"""Frozen-checkpoint inference screen: exact per-image counts, no training.

Val exports a fixed threshold grid. Test accepts only a separately frozen
selection file and never searches thresholds. Model source trees stay untouched.
"""
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


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(tp, pred, label):
    tp = np.asarray(tp, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    return dict(iou=div(tp, pred+label-tp), dice=div(2*tp, pred+label),
                precision=div(tp, pred), recall=div(tp, label))


def count_grid(prob, label, thresholds):
    outputs = []
    for start in range(0, len(thresholds), 8):
        ts = prob.new_tensor(thresholds[start:start+8])
        pred = prob[:, None] > ts[None, :, None, None]
        tp = (pred & label[:, None]).sum((2, 3), dtype=torch.int64)
        ps = pred.sum((2, 3), dtype=torch.int64)
        outputs.append(torch.stack((tp, ps), -1).cpu().numpy())
    return np.concatenate(outputs, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('validation', 'test'), required=True)
    p.add_argument('--selection', type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    plan = read(root/'plan.json')
    # Strict threshold equality and empty masks must match the historical metric.
    check_prob = torch.tensor([[[0., .5], [.50000006, 1.]], [[0., 0.], [0., 0.]]])
    check_label = torch.tensor([[[False, True], [False, True]], [[False, False], [False, False]]])
    check_counts = count_grid(check_prob, check_label, [.5, .7])
    assert check_counts.tolist() == [[[1,2],[1,1]],[[0,0],[0,0]]]
    check_metrics = metrics(check_counts[:,:,0], check_counts[:,:,1], np.array([[2],[0]]))
    assert np.array_equal(check_metrics['iou'], [[1/3, .5], [0., 0.]])
    source_sha = os.environ['RAPID_EVALUATION_COMMIT']
    assert len(source_sha) == 40
    run = root/a.split
    run.mkdir(exist_ok=True)
    assert not (run/'result.json').exists(), 'Never overwrite a completed export'
    manifest = read(root/'deployment.json')
    for name in ('evaluate.py', 'plan.json'):
        assert sha(root/name) == manifest['sha256'][name]
    assert not plan['ensemble_allowed'] and len(plan['sources']) == 1
    sources = plan['sources']
    modes = {'r2s1219': plan['threshold_grid']}
    selection_sha = None
    if a.split == 'test':
        assert a.selection and os.environ.get('RAPID_TEST_ALLOWED') == '1'
        selection = read(a.selection)
        assert selection['frozen'] and selection['plan_sha256'] == sha(root/'plan.json')
        assert selection['evaluation_source_git_commit'] == source_sha
        assert selection['validation_result_sha256'] == sha(root/'validation/result.json')
        assert selection['methods'], 'No candidate passed the declared Val gate'
        modes = {'r2s1219': [.5]}
        for name, item in selection['methods'].items():
            assert item['passed_validation_gate']
            modes.setdefault(name, [.5])
            modes[name] = sorted(set(modes[name]+[item['threshold']]))
        assert set(modes) == {'r2s1219'}
        selection_sha = sha(a.selection)
    repo = Path(sources[0]['repository'])
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from tools.export_validation_metrics import build_model, validation_loader
    import Config as config
    from training_recipe import recipe_metadata
    from Load_Dataset import ImageToImage2D, ValGenerator
    from utils import read_text
    from torch.utils.data import DataLoader
    assert config.experiment_name == 'r2_single_cosine' and not config.text_use_lora
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    models = []
    checkpoints = []
    for s in sources:
        actual = subprocess.check_output(['git', '-C', s['repository'], 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == s['source_git_commit']
        assert not subprocess.check_output(['git', '-C', s['repository'], 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        checkpoint = torch.load(s['checkpoint'], map_location='cpu', weights_only=True)
        assert checkpoint['source_git_commit'] == actual
        assert checkpoint['seed'] == s['seed'] and checkpoint['best_epoch'] == s['best_epoch']
        assert checkpoint['epochs'] == 80 and checkpoint['selection_metric'] == 'iou'
        assert checkpoint['experiment_name'] == config.experiment_name
        assert checkpoint['architecture_version'] == config.experiment_architecture_version
        assert checkpoint['training_recipe'] == recipe_metadata(config)
        assert not checkpoint['text_use_lora'] and checkpoint.get('boundary_loss_weight', 0) == 0
        model = build_model()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.requires_grad_(False)
        models.append(model.cuda().eval())
        cp = Path(s['checkpoint'])
        checkpoints.append(dict(source_git_commit=actual, path=str(cp), bytes=cp.stat().st_size,
                                mtime_ns=cp.stat().st_mtime_ns, sha256=sha(cp)))
        del checkpoint
    if a.split == 'validation':
        dataset, loader = validation_loader(16)
    else:
        text = read_text(os.path.join(config.test_dataset, 'Test_text.xlsx'))
        dataset = ImageToImage2D(config.test_dataset, config.task_name, text,
            ValGenerator(output_size=[224,224]), image_size=224)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    assert len(dataset) == plan[a.split+'_samples']
    names_all, labels_all = [], []
    counts = {k: [] for k in modes}
    individual_counts = {s['label']: [] for s in sources}
    started = time.time()
    first_batch_seconds = None
    with torch.inference_mode():
        for batch_index, (batch, names) in enumerate(loader):
            batch_start = time.time()
            inputs = [batch[k].cuda(non_blocking=True) for k in ('image', 'input_ids', 'attention_mask')]
            labels = batch['label'].bool().cuda(non_blocking=True)
            assert labels.ndim == 3 and labels.shape[-2:] == (224,224)
            for model, s in zip(models, sources):
                probability = model(*inputs)[:, 0].float()
                assert torch.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
                individual_counts[s['label']].append(count_grid(probability, labels, [.5]))
                if s['label'] == 'r2s1219':
                    counts['r2s1219'].append(count_grid(probability, labels, modes['r2s1219']))
            names_all.extend(map(str, names))
            labels_all.extend(labels.sum((1,2)).cpu().tolist())
            if batch_index == 0:
                first_batch_seconds = time.time()-batch_start
    assert len(names_all) == len(dataset) and len(set(names_all)) == len(dataset)
    label_array = np.asarray(labels_all, dtype=np.int64)[:, None]
    parity = {}
    for s in sources:
        name = s['label']
        reference = read(root/'references'/(name+'_'+a.split+'.json'))
        assert reference['checkpoint_git_commit'] == s['source_git_commit']
        expected = {r['name']: r for r in reference['records']}
        assert set(expected) == set(names_all)
        actual = np.concatenate(individual_counts[name], axis=0)[:, 0]
        for index, filename in enumerate(names_all):
            row = expected[filename]
            assert labels_all[index] == row['label_pixels']
            assert int(actual[index,1]) == row['prediction_pixels'], (name,filename,'prediction mismatch')
            assert int(actual[index,0]) == round(row['recall']*row['label_pixels']), (name,filename,'TP mismatch')
        parity[name] = dict(exact_per_image_counts=True, reference_sha256=sha(root/'references'/(name+'_'+a.split+'.json')))
    exports = {}
    for name, values in counts.items():
        array = np.concatenate(values, axis=0)
        ms = metrics(array[:,:,0], array[:,:,1], label_array)
        exports[name] = dict(thresholds=modes[name], tp=array[:,:,0].tolist(), pred=array[:,:,1].tolist(),
                             macro={k: v.mean(0).tolist() for k,v in ms.items()})
    for cp in checkpoints:
        path = Path(cp['path'])
        assert path.stat().st_size == cp['bytes'] and path.stat().st_mtime_ns == cp['mtime_ns']
    result = dict(completed=True, split=a.split, test_split_accessed=a.split=='test',
        evaluation_source_git_commit=source_sha, script_sha256=sha(__file__), plan_sha256=sha(root/'plan.json'),
        selection_sha256=selection_sha, samples=len(names_all), names=names_all, label_pixels=labels_all,
        modes=exports, checkpoint_provenance=checkpoints, parity=parity,
        started_unix=started, completed_unix=time.time(), first_batch_seconds=first_batch_seconds,
        inference_seconds=time.time()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        no_training=True, no_tta=True, no_component_filter=True)
    (run/'result.json').write_text(json.dumps(result)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('names','label_pixels','modes','checkpoint_provenance')}), flush=True)


if __name__ == '__main__':
    main()
