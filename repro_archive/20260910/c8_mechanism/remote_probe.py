"""Read-only checkpoint history extraction and fixed-Train gradient diagnosis.

Run on the original server. No optimizer, checkpoint writes, or Test loader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def sha_state(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def small(value):
    import torch
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else {
            'tensor_shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): small(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [small(v) for v in value]
    return value


def history(args, manifest):
    import torch
    for key, row in manifest.items():
        ck = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
        assert ck['source_git_commit'] == row['source_git_commit']
        assert ck['best_epoch'] == 80 and ck['seed'] == 1219
        out = {k: small(v) for k, v in ck.items()
               if k not in ('state_dict', 'optimizer') and 'rng' not in k}
        out['optimizer_param_groups'] = small(ck['optimizer']['param_groups'])
        out['checkpoint_path'] = row['checkpoint']
        out['state_sha256'] = sha_state(ck['state_dict'])
        out['runtime_git_commit'] = git(row['repo'], 'rev-parse', 'HEAD')
        out['runtime_git_status'] = git(row['repo'], 'status', '--porcelain')
        out['analysis_git_commit'] = args.analysis_sha
        save(Path(args.output) / (key + '_history.json'), out)
        print('Extracted history:', key, len(ck['epoch_history']), flush=True)
        del ck


def probe(args, manifest):
    row = manifest['c8']
    os.environ.update(BETTERLVIT_EXPERIMENT='c8_race_pe_v2_aux',
        HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='1219',
        TOKENIZERS_PARALLELISM='false', TEST_SPLIT_ALLOWED='0', AUTO_TEST_EVALUATE='0')
    os.chdir(row['repo'])
    sys.path[:0] = [row['repo'], str(Path(row['repo']) / 'tools')]
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    import Config as config
    from export_validation_metrics import build_model
    from Load_Dataset import ImageToImage2D, ValGenerator
    from race_pe_objective import RACEPEObjective, masked_mean
    from utils import read_text

    random.seed(1219); np.random.seed(1219); torch.manual_seed(1219)
    torch.cuda.manual_seed_all(1219)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    assert git(row['repo'], 'rev-parse', 'HEAD') == row['source_git_commit']
    runtime_status = git(row['repo'], 'status', '--porcelain')
    # The existing server dataset mount is an untracked symlink, not source.
    assert runtime_status in ('', '?? datasets'), runtime_status
    if runtime_status:
        assert (Path(row['repo'])/'datasets').is_symlink()
    dataset_mount = str((Path(row['repo'])/'datasets').resolve())
    assert not config.text_use_lora and not config.race_pe_route_enabled
    assert config.race_pe_v2_enabled and config.race_aux_weight == .05
    ck = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
    assert ck['source_git_commit'] == row['source_git_commit']
    model = build_model()
    model.load_state_dict(ck['state_dict'], strict=True)
    before = sha_state(model.state_dict())
    del ck
    model.cuda().eval()
    assert not any(p.requires_grad for p in model.text_encoder.parameters())
    criterion = RACEPEObjective(aux_weight=config.race_aux_weight,
        pixel_only=config.race_pe_pixel_only,
        drop_report_consistency=config.race_pe_v2_enabled,
        dice_weight=config.dice_loss_weight, focal_weight=config.focal_loss_weight,
        focal_gamma=config.focal_gamma,
        focal_positive_weight=config.focal_positive_weight,
        focal_negative_weight=config.focal_negative_weight).cuda()
    dataset = ImageToImage2D(config.train_dataset, config.task_name,
        read_text(os.path.join(config.task_dataset, 'Train_Val_text.xlsx')),
        ValGenerator([config.img_size, config.img_size]), image_size=config.img_size)
    indices = sorted(range(len(dataset)), key=lambda i: hashlib.sha256(
        ('c8-mechanism-v1:' + dataset.mask_list[i]).encode()).hexdigest())[:32]
    names = [dataset.mask_list[i] for i in indices]
    loader = DataLoader(Subset(dataset, indices), batch_size=8,
                        shuffle=False, num_workers=0)
    features = {}
    hooks = [getattr(model, name).register_forward_hook(
        lambda module, inputs, output, name=name: features.__setitem__(name, output))
        for name in ('inc', 'down1', 'down2', 'down3')]
    result = dict(analysis_git_commit=args.analysis_sha,
        c8_training_sha=row['source_git_commit'], checkpoint=row['checkpoint'],
        runtime_git_status=runtime_status, dataset_mount=dataset_mount,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='C8 Best, fixed 32 Train images, eval mode, batch 8, no optimizer',
        samples=names, seed=1219, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), batches=[])
    start = time.time()
    for batch_index, (sample, filenames) in enumerate(loader):
        sample = {k: v.cuda() for k, v in sample.items()}
        args_forward = (sample['image'], sample['input_ids'], sample['attention_mask'])
        kw = dict(race_slot_targets=sample['race_slot_targets'],
                  race_zone_basis=sample['race_zone_basis'])
        outputs = model(*args_forward, return_aux=True, **kw)
        taps = [features[k] for k in ('inc', 'down1', 'down2', 'down3')]
        target = sample['label'].float()
        mask = target.unsqueeze(1) if target.ndim == 3 else target
        pixel, presence, occupancy = [], [], []
        for route in outputs['pe_routes']:
            gt = F.adaptive_avg_pool2d(mask, route['extent_logits'].shape[-2:])
            mass = route['basis'].sum((2, 3)); valid = mass > 0
            area = (gt * route['basis']).sum((2, 3)) / mass.clamp_min(1)
            pixel.append(F.binary_cross_entropy_with_logits(route['extent_logits'], gt))
            presence.append(masked_mean(F.binary_cross_entropy_with_logits(
                route['presence_logits'], (area > 0).float(), reduction='none'), valid))
            occupancy.append(masked_mean((route['occupancy']-area).square(), valid))
        slots, labels = outputs['slot_logits'], outputs['race_slot_targets']
        text = masked_mean(F.binary_cross_entropy_with_logits(slots[:, :6],
            labels[:, :6].clamp_min(0), reduction='none'), labels[:, :6] >= 0)
        count = masked_mean(F.cross_entropy(slots[:, 6:], labels[:, 6:].argmax(1),
            reduction='none'), (labels[:, 6:] >= 0).all(1))
        losses = dict(main=criterion.segmentation(outputs['final'], target),
            pixel=.02*torch.stack(pixel).mean(),
            presence=.01*torch.stack(presence).mean(),
            occupancy=.005*torch.stack(occupancy).mean(),
            text=.0075*text, count=.0025*count)
        total = criterion(outputs, target)
        assert torch.allclose(total, sum(losses.values()), atol=1e-7, rtol=1e-6)
        losses['visual_aux'] = losses['pixel']+losses['presence']+losses['occupancy']
        scalar_losses = {name: loss.item() for name, loss in losses.items()}
        gradients = {name: torch.autograd.grad(loss, taps, allow_unused=True,
            retain_graph=True) for name, loss in losses.items()}
        metrics = {}
        for name, grads in gradients.items():
            layer_metrics = []
            for g, ref in zip(grads, gradients['main']):
                if g is None:
                    layer_metrics.append(dict(connected=False, norm=0., ratio=0., cosine=None))
                else:
                    norm = torch.linalg.vector_norm(g).item()
                    refnorm = torch.linalg.vector_norm(ref).item()
                    dot = (g*ref).sum().item()
                    layer_metrics.append(dict(connected=True, norm=norm,
                        main_norm=refnorm, ratio=norm/refnorm,
                        cosine=dot/(norm*refnorm) if norm*refnorm else None))
            metrics[name] = layer_metrics
        slot_grads = torch.autograd.grad(losses['text']+losses['count'],
            tuple(model.race.slot_head.parameters()), allow_unused=True, retain_graph=True)
        slot_norm = sum(g.square().sum().item() for g in slot_grads if g is not None)**.5
        assert slot_norm > 0
        assert all(g is None for name in ('text', 'count') for g in gradients[name])
        # Test exact output invariance; hooks may change features only after diagnosis.
        expected = outputs['final'].detach().clone()
        del gradients, taps, losses, outputs, total
        features.clear()
        model.race_enabled = False
        with torch.no_grad():
            disabled = model(*args_forward, **kw)
        model.race_enabled = True
        difference = (disabled-expected).abs().max().item()
        assert torch.equal(disabled, expected)
        result['batches'].append(dict(batch=batch_index, images=list(filenames),
            gradients=metrics, weighted_losses=scalar_losses, slot_head_gradient_norm=slot_norm,
            inference_max_abs_diff=difference))
        print('Gradient batch complete:', batch_index+1, '/ 4', flush=True)
        features.clear()
        del sample, args_forward, kw, disabled, expected, slot_grads
    for hook in hooks:
        hook.remove()
    after = sha_state(model.state_dict())
    assert before == after
    result.update(state_sha256_before=before, state_sha256_after=after,
        unchanged_weights_and_buffers=True, runtime_seconds=time.time()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    save(Path(args.output)/'c8_gradient_probe.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('history', 'probe'))
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--analysis-sha', required=True)
    args = parser.parse_args()
    assert len(args.analysis_sha) == 40
    manifest = json.loads(Path(args.manifest).read_text())
    (history if args.mode == 'history' else probe)(args, manifest)


if __name__ == '__main__':
    main()
