"""Check continuation reporting for a new best, retained epoch-80 best and invalid pairs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = dict(split='test', samples=2, seed=1219, threshold=0.5,
            selection_metric='iou', architecture_version='p11_dual_grain_visual_28_v1',
            experiment='p11_dual_grain', epochs=80, checkpoint_best_epoch=80,
            checkpoint_git_commit='2fc6ab5c8e4662d741fd8b994e55b780391948ac',
            records=[dict(name=str(i), label_pixels=10, iou=0.5, dice=2/3,
                          precision=0.8, recall=0.7) for i in range(2)])
        (root / 'parent.json').write_text(json.dumps(parent))

        def run(candidate, best, success):
            (root / 'candidate.json').write_text(json.dumps(candidate))
            (root / 'selection.json').write_text(json.dumps(dict(completed_epochs=150, best_epoch=best)))
            command = [sys.executable, str(Path(__file__).with_name('compare_p11_continuation.py'))]
            for key in ('parent', 'candidate', 'selection', 'output'):
                command += ['--' + key, str(root / (key + '.json'))]
            result = subprocess.run(command, capture_output=True, text=True)
            assert (result.returncode == 0) == success, result.stderr
            return json.loads((root / 'output.json').read_text()) if success else None

        retained = run(parent, 80, True)
        assert retained['metrics']['iou']['delta'] == 0
        candidate = json.loads(json.dumps(parent))
        candidate.update(epochs=150, checkpoint_best_epoch=130)
        for row in candidate['records']:
            row['iou'] += 0.1
        improved = run(candidate, 130, True)
        assert abs(improved['metrics']['iou']['delta'] - 0.1) < 1e-12
        candidate['threshold'] = 0.6
        run(candidate, 130, False)
        candidate['threshold'] = 0.5
        candidate['records'][0]['name'] = 'wrong-case'
        run(candidate, 130, False)
    print('PASS: new best, retained best, threshold mismatch, case mismatch')


if __name__ == '__main__':
    main()
