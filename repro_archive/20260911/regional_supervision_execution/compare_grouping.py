"""RS3 versus RS2 attribution diagnostic using the frozen paired bootstrap.

The plan requires a positive IoU paired interval to attribute a grouping gain;
the separate +0.001 region-versus-RS1 gate is not imposed on this comparison.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from remote_ops import HERE, save


def main():
    source=json.loads((HERE/'sources.json').read_text())['rs3']
    script=Path(source['local_repository'])/'tools/compare_regional.py'
    spec=importlib.util.spec_from_file_location('frozen_regional_comparison',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=[HERE/'rs2_results/validation.json',HERE/'rs3_results/validation.json']
    a,b=[json.loads(path.read_text()) for path in paths]
    assert a['experiment']=='rs2_local_iou' and b['experiment']=='rs3_balanced_iou'
    comparison=module.compare(a,b,regional_increment=True)
    result={k:comparison[k] for k in ('control','candidate','seed','control_sha','candidate_sha',
        'deltas','small_area_cutoff','small_count','split','test_split_accessed','stable_gain_proven')}
    result.update(comparison_role='grouping_attribution_only',
        grouping_iou_ci_positive=comparison['deltas']['iou']['ci95'][0]>0,
        gate_note='Positive paired IoU CI is required for grouping attribution; R2 and RS1 gates remain separate.',
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        comparison_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    save('rs3_results/rs2_vs_rs3.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
