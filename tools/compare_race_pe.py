"""IoU-primary validation gate; historical RACE gates stay unchanged."""
import json
import subprocess
import sys
from pathlib import Path


def passes(result):
    overall = result['overall']
    smallest = result['lesion_size_quartiles'][0]
    return (overall['iou']['mean_delta'] >= .003
        and overall['dice']['mean_delta'] >= 0
        and overall['precision']['mean_delta'] >= 0
        and smallest['dice']['mean_delta'] >= 0
        and smallest['recall']['mean_delta'] >= 0
        and overall['brier_lower_is_better']['mean_delta'] <= 0)


if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).with_name(
        'compare_validation_metrics.py')), *sys.argv[1:]], check=True)
    path = Path(sys.argv[sys.argv.index('--output') + 1])
    result = json.loads(path.read_text())
    result['historical_dice_gate'] = result.pop('passes_numeric_screen')
    result['race_pe_gate'] = {
        'primary_metric': 'macro_iou', 'minimum_delta': .003,
        'passes_single_seed_screen': passes(result),
        'iou_bootstrap_ci_positive': result['overall']['iou']['bootstrap_95_ci'][0] > 0,
        'stable_gain_proven': False,
        'test_authorized': False,
        'next_stage': 'Review diagnostics, then preregister paired additional seeds; no automatic extension',
    }
    path.write_text(json.dumps(result, indent=2) + '\n')
