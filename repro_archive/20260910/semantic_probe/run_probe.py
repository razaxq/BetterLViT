"""Frozen R2 Train/Val probe. No Test reads, backbone updates or feature files."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / 'manifest.json').read_text())
ROOT = Path(MANIFEST['baseline_repository'])
os.environ.update(BETTERLVIT_EXPERIMENT='r2_single_cosine', TEST_SPLIT_ALLOWED='0', AUTO_TEST_EVALUATE='0',
                  HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  PYTHONHASHSEED='1219', TOKENIZERS_PARALLELISM='false')


def guard(event, args):
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
        parts = Path(os.fsdecode(args[0])).parts
        if 'Test_Folder' in parts or 'Test_text.xlsx' in parts:
            raise RuntimeError('Test access is prohibited in this diagnostic')


sys.addaudithook(guard)
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
os.chdir(ROOT)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import Config as config
from Load_Dataset import ImageToImage2D, ValGenerator
from export_validation_metrics import build_model, validation_loader
from utils import read_text, WeightedDiceFocal
sys.path.insert(0, str(HERE))
from analysis import SELECTORS, blocks, digest, metrics, random_scores, selected_mask, summarize, write_json


def deterministic():
    torch.manual_seed(1219); np.random.seed(1219)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Features:
    def __init__(self, model):
        self.model, self.values = model, {}
        self.handles = []
        for name, module in [('fine', model.down1), ('coarse', model.reconstruct2), ('logit', model.outc)]:
            def hook(mod, inp, out, key=name):
                self.values[key] = F.avg_pool2d(out, 2).detach() if key != 'logit' else out.detach()
            self.handles.append(module.register_forward_hook(hook))

    def batch(self, batch):
        self.values.clear()
        with torch.inference_mode():
            p = self.model(batch['image'].cuda(), batch['input_ids'].cuda(), batch['attention_mask'].cuda())
            assert self.values['fine'].shape[1:] == (128, 56, 56)
            assert self.values['coarse'].shape == self.values['fine'].shape
            assert torch.equal(p, self.values['logit'].sigmoid())
            return dict(fine=self.values['fine'].cpu().half(), coarse=self.values['coarse'].cpu().half(),
                        logit=self.values['logit'].cpu(), probability=p.cpu(), label=batch['label'].cpu().byte())


def collect(extractor, loader):
    chunks, names = [], []
    for batch, ns in loader:
        chunks.append(extractor.batch(batch)); names.extend(ns)
    return {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}, names


def moments(x):
    total = torch.zeros(x.shape[1], dtype=torch.float64)
    squares = total.clone(); count = 0
    for b in x.split(16):
        b = b.double(); total += b.sum((0,2,3)); squares += b.square().sum((0,2,3))
        count += b.shape[0] * b.shape[2] * b.shape[3]
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    return mean.float()[None,:,None,None], std.float()[None,:,None,None]


class Heads(nn.Module):
    def __init__(self, norm):
        super().__init__()
        for k, v in norm.items(): self.register_buffer(k, v)
        self.fine = nn.Conv2d(128, 1, 1)
        self.coarse = nn.Conv2d(128, 1, 1)
        self.local = nn.Sequential(nn.Conv2d(257, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.local[-1].weight); nn.init.zeros_(self.local[-1].bias)

    def forward(self, batch):
        fine = (batch['fine'] - self.fine_mean) / self.fine_std
        coarse = (batch['coarse'] - self.coarse_mean) / self.coarse_std
        logits = batch['logit']
        fine_p = F.interpolate(self.fine(fine), size=(224,224), mode='nearest').sigmoid()
        coarse_p = F.interpolate(self.coarse(coarse), size=(224,224), mode='nearest').sigmoid()
        raw = self.local(torch.cat((fine, coarse, F.avg_pool2d(logits, 4)), 1))
        delta = F.interpolate(.5 * raw.tanh(), size=(224,224), mode='nearest')
        return dict(fine_probe=fine_p, coarse_probe=coarse_p, correction=(logits + delta).sigmoid())


def gpu_batch(cache, indices):
    return {k: v[indices].float().cuda() for k,v in cache.items()}


def fit(cache, steps=256):
    norm = {}
    for k in ('fine', 'coarse'):
        norm[k+'_mean'], norm[k+'_std'] = moments(cache[k])
    torch.manual_seed(1219)
    heads = Heads(norm).cuda().train()
    opt = torch.optim.Adam(heads.parameters(), lr=1e-3, weight_decay=1e-4)
    objective = WeightedDiceFocal(dice_weight=.5, focal_weight=.5)
    rng = np.random.default_rng(1219)
    first = gpu_batch(cache, np.arange(min(16, len(cache['label']))))
    initial = heads(first)
    assert torch.equal(initial['correction'], first['probability']), 'Zero correction must reproduce R2'
    history = []
    torch.cuda.synchronize(); start = time.time()
    for step in range(steps):
        batch = gpu_batch(cache, rng.integers(0, len(cache['label']), size=16))
        lr = 1e-5 + .5 * (1e-3-1e-5) * (1 + math.cos(math.pi * step / max(1, steps-1)))
        for group in opt.param_groups: group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        outputs = heads(batch)
        losses = [objective(outputs[k], batch['label']) for k in ('fine_probe', 'coarse_probe', 'correction')]
        loss = sum(losses)
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
        history.append(dict(step=step+1, lr=lr, losses=[float(v.detach()) for v in losses]))
    torch.cuda.synchronize()
    return heads.eval(), history, (time.time()-start)/steps


def case_record(name, target, original, outputs, texture):
    target, pred = target.astype(bool), original > .5
    fp, fn = pred & ~target, ~pred & target
    error = fp | fn; corrected = outputs['correction'] > .5
    scores = dict(disagreement=blocks(np.abs(outputs['fine_probe']-outputs['coarse_probe'])).mean(1),
                  uncertainty=blocks(4 * original * (1-original)).mean(1),
                  texture=blocks(texture).mean(1), random=random_scores(name))
    row = dict(name=name, baseline=metrics(pred,target), fine_probe=metrics(outputs['fine_probe']>.5,target),
               coarse_probe=metrics(outputs['coarse_probe']>.5,target), full_correction=metrics(corrected,target), selectors={})
    for key in SELECTORS:
        mask, indices = selected_mask(scores[key], MANIFEST['selected_blocks'])
        mixed = np.where(mask, corrected, pred)
        oracle = np.where(mask, target, pred)
        cap = int((error & mask).sum())
        row['selectors'][key] = dict(blocks=indices.tolist(), fp_captured=int((fp & mask).sum()),
            fn_captured=int((fn & mask).sum()), error_capture=cap/int(error.sum()) if error.any() else None,
            corrected=metrics(mixed,target), oracle_iou=metrics(oracle,target)['iou'])
    patch = dict(**scores, fp=blocks(fp).sum(1), fn=blocks(fn).sum(1))
    return row, patch


def evaluate(extractor, heads, loader):
    records, patch_rows = [], []
    for batch, names in loader:
        cache = extractor.batch(batch)
        with torch.no_grad():
            out = {k: v[:,0].cpu().numpy() for k,v in heads(gpu_batch(cache, np.arange(len(names)))).items()}
        gray = batch['image'].mean(1, keepdim=True)
        texture = (gray - F.avg_pool2d(gray, 5, 1, 2)).abs()[:,0].numpy()
        for i, name in enumerate(names):
            row, patches = case_record(name, batch['label'][i].numpy(), cache['probability'][i,0].numpy(),
                                       {k:v[i] for k,v in out.items()}, texture[i])
            records.append(row); patch_rows.append(patches)
    return records, {k: np.stack([p[k] for p in patch_rows]) for k in patch_rows[0]}


def main(args):
    started = time.time(); deterministic()
    assert not MANIFEST['test_split_allowed']
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip() == MANIFEST['baseline_source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=ROOT, text=True).strip()
    assert not config.text_use_lora and config.boundary_loss_weight == 0
    assert digest(HERE/'r2_validation.json') == MANIFEST['validation_sha256']
    checkpoint = torch.load(MANIFEST['baseline_checkpoint'], map_location='cpu', weights_only=True)
    assert checkpoint['source_git_commit'] == MANIFEST['baseline_source_git_commit']
    model = build_model(); model.load_state_dict(checkpoint['state_dict'], strict=True); del checkpoint
    model.requires_grad_(False); model.cuda().eval()
    assert not any(p.requires_grad for p in model.parameters())
    model_hash = state_hash(model)
    extractor = Features(model)
    text = read_text(os.path.join(config.task_dataset, 'Train_Val_text.xlsx'))
    train = ImageToImage2D(config.train_dataset, config.task_name, text,
                          ValGenerator([224,224]), image_size=224)
    assert Path(train.dataset_path).resolve().name == 'Train_Folder'
    ordered = sorted(range(len(train)), key=lambda i: hashlib.sha256(('semantic-probe-v1:'+train.mask_list[i]).encode()).hexdigest())
    fit_ix = ordered[:MANIFEST['fit_samples']]; audit_ix = ordered[MANIFEST['fit_samples']:MANIFEST['fit_samples']+MANIFEST['audit_samples']]
    assert len(fit_ix)==512 and len(audit_ix)==128 and not set(fit_ix)&set(audit_ix)
    def loader(indices): return DataLoader(Subset(train, indices), batch_size=16, shuffle=False, num_workers=0)
    torch.cuda.synchronize(); feature_start = time.time()
    cache, names = collect(extractor, loader(fit_ix[:16] if args.preflight else fit_ix))
    torch.cuda.synchronize(); per_image = (time.time()-feature_start)/len(names)
    if args.preflight:
        heads1, hist1, step_seconds = fit(cache, 8)
        hash1 = state_hash(heads1); del heads1
        heads2, hist2, _ = fit(cache, 8)
        assert state_hash(heads2) == hash1 and hist1 == hist2
        assert state_hash(model) == model_hash and all(p.grad is None for p in model.parameters())
        predicted = (time.time()-started) + 1.5 * per_image * (512+128+1429) + 1.5 * step_seconds * 256 + 120
        result = dict(status='ok', mode='preflight', baseline_state_unchanged=True, deterministic_two_repeats=True,
                      probe_state_hash=hash1, loss_history=hist1, feature_seconds_per_image=per_image,
                      head_seconds_per_step=step_seconds, predicted_run_seconds=predicted,
                      tested_train_names=names, test_split_accessed=False,
                      scripts_sha256={p.name:digest(p) for p in HERE.glob('*.py')})
        write_json(args.output, result); print(json.dumps(result)); return
    heads, history, step_seconds = fit(cache, MANIFEST['steps'])
    del cache
    write_json(args.output/'probe_training.json', dict(fit_names=names, history=history, seconds_per_step=step_seconds))
    audit, _ = evaluate(extractor, heads, loader(audit_ix))
    write_json(args.output/'audit_records.json', dict(split='train_probe_audit', records=audit))
    dataset, val_loader = validation_loader(16)
    assert len(dataset) == 1429 and Path(dataset.dataset_path).resolve().name == 'Val_Folder'
    records, patch_data = evaluate(extractor, heads, val_loader)
    reference = json.loads((HERE/'r2_validation.json').read_text())
    refs = {r['name']: r for r in reference['records']}
    assert set(refs) == {r['name'] for r in records}
    maximum_error = 0.
    for r in records:
        for metric in ('iou','dice','precision','recall'):
            error = abs(r['baseline'][metric] - refs[r['name']][metric])
            maximum_error = max(maximum_error,error)
            assert error < 1e-12, (r['name'],metric,error)
    assert state_hash(model) == model_hash and all(p.grad is None for p in model.parameters())
    metadata = dict(split='validation', test_split_accessed=False,
                    diagnostic_source_git_commit=args.source_sha, baseline_source_git_commit=MANIFEST['baseline_source_git_commit'],
                    checkpoint_sha256=digest(MANIFEST['baseline_checkpoint']), baseline_state_sha256=model_hash,
                    baseline_state_unchanged=True, baseline_per_image_max_difference=maximum_error,
                    checkpoint_best_epoch=MANIFEST['baseline_best_epoch'], frozen_model=True)
    write_json(args.output/'records.json', dict(**metadata, records=records))
    np.savez_compressed(args.output/'patch_scores.npz', names=np.asarray([r['name'] for r in records]), **patch_data)
    torch.save(dict(state_dict=heads.cpu().state_dict(), metadata=metadata), args.output/'diagnostic_heads.pt')
    result = summarize(records); write_json(args.output/'summary.json', result)
    print(json.dumps(dict(status='complete', proceed_to_architecture=result['proceed_to_architecture'],
                          baseline_per_image_max_difference=maximum_error, elapsed_seconds=time.time()-started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    assert len(args.source_sha) == 40 and all(c in '0123456789abcdef' for c in args.source_sha)
    if args.preflight:
        main(args)
    else:
        args.output.mkdir(exist_ok=False)
        start = time.time()
        runtime = dict(phase='running', submitted_unix=start, diagnostic_source_git_commit=args.source_sha,
                       baseline_source_git_commit=MANIFEST['baseline_source_git_commit'], test_split_accessed=False)
        write_json(args.output/'runtime.json',runtime)
        try:
            main(args)
            runtime.update(phase='complete', completed_unix=time.time(),
                           artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.is_file() and p.name!='runtime.json'})
        except BaseException:
            runtime.update(phase='failed', completed_unix=time.time(), traceback=traceback.format_exc())
            raise
        finally:
            write_json(args.output/'runtime.json',runtime)
