"""Frozen T2 interventions. Only the added adapter sees mismatched text."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXPECTED = '488ef093de80df71ee77741a8c7ee7b938c7d6b5'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def donor_indices(names, ids, masks):
    keys = [tuple(row.tolist()) for row in ids]
    buckets = {}
    for i, length in enumerate(masks.sum(1).tolist()):
        buckets.setdefault(int(length), []).append(i)
    result = list(range(len(names)))
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda i: names[i])
        for pos, i in enumerate(ordered):
            for shift in range(1, len(ordered)):
                j = ordered[(pos + shift) % len(ordered)]
                if keys[i] != keys[j]:
                    result[i] = j
                    break
    return result

def records(prob, labels, names):
    prob = prob[:, 0].float().cpu()
    label = labels.bool().cpu()
    assert label.shape == prob.shape
    assert torch.isfinite(prob).all()
    pred = prob > .5
    tp = (pred & label).sum((1, 2))
    pp, gp = pred.sum((1, 2)), label.sum((1, 2))
    rows = []
    for i, name in enumerate(names):
        t, p, g = int(tp[i]), int(pp[i]), int(gp[i])
        rows.append(dict(name=str(name), tp=t, prediction_pixels=p, label_pixels=g,
            fp=p-t, fn=g-t, iou=t/(p+g-t) if p+g-t else 0.,
            dice=2*t/(p+g) if p+g else 0., precision=t/p if p else 0.,
            recall=t/g if g else 0., brier=float((prob[i]-label[i].float()).square().mean(dtype=torch.float64))))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repository', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    start = time.time()
    root = Path(a.repository).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'tools'))
    # The exporter has an import-time parser; it must not parse this program's args.
    sys.argv = [sys.argv[0]]
    import export_validation_metrics as ex
    from nets.decoder_context import adapter_metadata
    from training_recipe import recipe_metadata
    assert ex.git_commit() == EXPECTED
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'], text=True).strip()
    reference = json.loads(Path(a.reference).read_text())
    ckpt = Path(reference['checkpoint'])
    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=True)
    assert checkpoint['source_git_commit'] == reference['checkpoint_git_commit'] == EXPECTED
    assert checkpoint['architecture_version'] == ex.config.experiment_architecture_version
    assert checkpoint['experiment_name'] == 't2_decoder_text'
    assert checkpoint['selection_metric'] == 'iou'
    assert checkpoint['decoder_context'] == adapter_metadata('text')
    assert checkpoint['training_recipe'] == recipe_metadata(ex.config)
    assert not checkpoint['text_use_lora'] and checkpoint.get('boundary_loss_weight', 0.) == 0.
    torch.manual_seed(1219)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = ex.build_model()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model = model.cuda().eval()
    before = model_digest(model)
    dataset, loader = ex.validation_loader(16)
    assert len(dataset) == 1429
    names = [str(n) for n in dataset.mask_list]
    lookup = {n:i for i,n in enumerate(names)}
    assert len(lookup) == len(names)
    donors = donor_indices(names, dataset.input_ids, dataset.attention_masks)
    assert all(int(dataset.attention_masks[i].sum()) == int(dataset.attention_masks[j].sum()) for i,j in enumerate(donors))
    donor_map = [dict(name=n, donor=names[donors[i]], active_tokens=int(dataset.attention_masks[i].sum()), changed=donors[i]!=i) for i,n in enumerate(names)]
    cache = []
    output = {mode:[] for mode in ('real', 'off', 'mismatch')}
    diagnostics = []
    first = None
    with torch.inference_mode():
        for pos in range(0, len(names), 16):
            t = model.encode_text(dataset.input_ids[pos:pos+16].cuda(), dataset.attention_masks[pos:pos+16].cuda())
            t = model.text_module4(t.transpose(1,2)).transpose(1,2)
            t = model.text_module3(t.transpose(1,2)).transpose(1,2)
            t = model.text_module2(t.transpose(1,2)).transpose(1,2)
            cache.append(t.cpu())
        cache = torch.cat(cache)
        for batch_index, (batch, bn) in enumerate(loader):
            args = (batch['image'].cuda(), batch['input_ids'].cuda(), batch['attention_mask'].cuda())
            stats = {}
            def observe(module, inputs, value):
                delta = value-inputs[0]
                stats['feature_rms'] = inputs[0].square().mean((1,2,3)).sqrt().cpu().tolist()
                stats['residual_rms'] = delta.square().mean((1,2,3)).sqrt().cpu().tolist()
            hook = model.decoder_context.register_forward_hook(observe)
            real = model(*args)
            hook.remove()
            if first is None:
                first = (tuple(x.clone() for x in args), real.clone())
                cached = cache[[lookup[str(n)] for n in bn]].cuda()
                def same_text(module, inputs):
                    assert torch.equal(cached, inputs[1]), 'cached text path changed'
                    return inputs
                check = model.decoder_context.register_forward_pre_hook(same_text)
                assert torch.equal(real, model(*args))
                check.remove()
            output['real'].extend(records(real, batch['label'], bn))
            hook = model.decoder_context.register_forward_hook(lambda module, inputs, value: inputs[0])
            off = model(*args)
            hook.remove()
            output['off'].extend(records(off, batch['label'], bn))
            ix = [donors[lookup[str(n)]] for n in bn]
            dt, dm = cache[ix].cuda(), dataset.attention_masks[ix].cuda()
            hook = model.decoder_context.register_forward_pre_hook(lambda module, inputs: (inputs[0],dt,dm))
            wrong = model(*args)
            hook.remove()
            output['mismatch'].extend(records(wrong, batch['label'], bn))
            for i, n in enumerate(bn):
                diagnostics.append(dict(name=str(n),feature_rms=stats['feature_rms'][i],residual_rms=stats['residual_rms'][i]))
            if batch_index % 15 == 0:
                print(json.dumps(dict(batch=batch_index+1,samples=len(output['real']),elapsed_seconds=time.time()-start)), flush=True)
        assert torch.equal(first[1],model(*first[0])), 'repeat output changed'
    assert model_digest(model) == before, 'model state changed'
    old = {r['name']:r for r in reference['records']}
    assert set(old) == set(lookup)
    for row in output['real']:
        prior = old[row['name']]
        assert row['prediction_pixels'] == prior['prediction_pixels'] and row['label_pixels'] == prior['label_pixels']
        reconstructed_tp = round(prior['dice']*(prior['prediction_pixels']+prior['label_pixels'])/2)
        assert row['tp'] == reconstructed_tp
        assert abs(row['iou']-prior['iou']) < 1e-12
    summary = {mode:{m:float(np.mean([r[m] for r in rows])) for m in ('iou','dice','precision','recall','brier')} for mode, rows in output.items()}
    result = dict(status='complete',split='validation',test_split_accessed=False,training_performed=False,
        source_git_commit=EXPECTED,analysis_git_commit=os.environ['ANALYSIS_GIT_COMMIT'],
        analysis_script_sha256=digest(__file__),reference_sha256=digest(a.reference),checkpoint_sha256=digest(ckpt),
        model_state_sha256=before,model_state_unchanged=True,repeat_output_exact=True,historical_counts_exact=True,
        threshold=.5,batch_size=16,samples=len(names),donor_map=donor_map,diagnostics=diagnostics,
        summaries=summary,per_mode=output,elapsed_seconds=time.time()-start,
        completed_unix=time.time(),torch_version=torch.__version__,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',summaries=summary,elapsed_seconds=result['elapsed_seconds'])),flush=True)

if __name__ == '__main__':
    main()
