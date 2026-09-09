"""Archive already-completed P11/P12 runs without re-evaluation or training polling."""
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import hf_xet
import torch

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    plan = json.loads((ROOT / 'additional_plan.json').read_text())
    manifests = []
    for arm in plan:
        run, repo = Path(arm['run']), Path(arm['repository'])
        input_checks = {}
        for filename, expected in arm['archived_input_canonical_sha256'].items():
            parsed = json.loads((run / filename).read_text())
            canonical = json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            assert hashlib.sha256(canonical).hexdigest() == expected, (arm['label'], filename)
            input_checks[filename] = dict(canonical_sha256=expected, server_file_sha256=sha256(run / filename),
                                         archived_file_sha256=arm['archived_input_sha256'][filename])
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == arm['source_git_commit']
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip()
        result = json.loads((run / arm['result_file']).read_text())
        assert result['split'] == arm['split'] and result['samples'] == arm['samples'] and result['threshold'] == .5
        assert result['checkpoint_git_commit'] == arm['best_source_git_commit']
        best = Path(result['checkpoint'])
        if arm['label'] == 'p11_150':
            selection = json.loads((run / 'selection.json').read_text())
            assert selection['completed_epochs'] == 150 and selection['best_epoch'] == 80
            assert Path(selection['checkpoint']) == best and sha256(best) == selection['checkpoint_sha256']
            last_candidates = list((repo / 'Covid19/BetterLViT/p11_dual_grain').glob('*/models/last_model-BetterLViT.pth.tar'))
            assert len(last_candidates) == 1
            last = last_candidates[0]
        else:
            last = best.with_name('last_model-BetterLViT.pth.tar')
        checks = []
        for model, expected_source, expected_epochs in ((best, arm['best_source_git_commit'], 80),
                                                       (last, arm['source_git_commit'], arm['epochs'])):
            checkpoint = torch.load(model, map_location='cpu', weights_only=True)
            assert checkpoint['state_dict'] and checkpoint['source_git_commit'] == expected_source
            assert checkpoint['epochs'] == expected_epochs and checkpoint['best_epoch'] == result['checkpoint_best_epoch']
            assert checkpoint['seed'] == 1219 and checkpoint['selection_metric'] == 'iou'
            assert not checkpoint['text_use_lora'] and checkpoint['boundary_loss_weight'] == 0
            checks.append(dict(path=str(model), source_git_commit=expected_source, epochs=expected_epochs,
                               best_epoch=checkpoint['best_epoch'], epoch=checkpoint.get('epoch')))
            del checkpoint
            gc.collect()
        session = last.parent.parent
        logs = list(session.glob('*.log'))
        events = list((session / 'tensorboard_logs').glob('events.out*'))
        assert len(logs) == len(events) == 1
        mapping = {logs[0].name: logs[0], arm['result_archive_name']: run / arm['result_file'],
                   'models/' + best.name: best, 'models/' + last.name: last,
                   'tensorboard_logs/' + events[0].name: events[0]}
        if arm['label'] == 'p11_150':
            mapping.update({'runtime.json': run / 'runtime.json', 'selection.json': run / 'selection.json'})
        hashes = hf_xet.hash_files([str(p) for p in mapping.values()])
        manifest = dict(label=arm['label'], source_git_commit=arm['source_git_commit'],
                        bucket_prefix=arm['source_git_commit'][:8], classification=arm['classification'],
                        training_completed_epochs=arm['epochs'], evaluation_split=arm['split'],
                        checkpoint_checks=checks, input_json_checks=input_checks,
                        files=[dict(source=str(src), remote=arm['source_git_commit'][:8] + '/' + name,
                                    bytes=info.file_size, xet_hash=info.hash)
                               for (name, src), info in zip(mapping.items(), hashes)])
        if arm['label'] == 'p11_150':
            manifest['best_checkpoint_parent_source_git_commit'] = arm['best_source_git_commit']
            next(f for f in manifest['files'] if f['remote'].endswith('/models/best_model-BetterLViT.pth.tar'))['copy_from'] = '2fc6ab5c/models/best_model-BetterLViT.pth.tar'
        (ROOT / (manifest['bucket_prefix'] + '_manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
        manifests.append(manifest)
    print(json.dumps(manifests))


if __name__ == '__main__':
    main()
