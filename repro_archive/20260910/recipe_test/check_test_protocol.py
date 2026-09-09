"""Offline checks: historical metric kernel identity and fail-closed Test provenance."""
import ast
import copy
import json
import tempfile
from pathlib import Path
from protocol import load_plan, sha256, validate_result, write_json

HERE = Path(__file__).resolve().parent


def must_reject(call):
    try:
        call()
    except (AssertionError, KeyError):
        return
    raise AssertionError('Invalid protocol was accepted')


def main():
    original = Path('D:/BetterLViT/outputs/race_pe_test_20260907/evaluate_test.py')
    assert sha256(original) == '2a4bdfa291c9b62789ef04f88e52c6610ee99b87626c8c9263f6264b5c1b0ab7'
    trees = [ast.parse(p.read_text(encoding='utf-8')) for p in (original, HERE / 'evaluate_test.py')]
    scientific = ('mask_boundary', 'boundary_f1', 'image_frequency_scores', 'build_model', 'test_loader')
    for name in scientific:
        nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for tree in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    loops = [next(n for n in ast.walk(tree) if isinstance(n, ast.With) and 'torch.inference_mode()' in ast.unparse(n.items[0])) for tree in trees]
    assert ast.dump(loops[0]) == ast.dump(loops[1])
    plan = load_plan(HERE)
    arm = plan['arms'][0]
    result = dict(split='test', test_split_accessed=True, user_authorized_test=True, threshold_selected_on_test=False,
                  seed=arm['seed'], experiment=arm['profile'], checkpoint=arm['checkpoint'],
                  checkpoint_git_commit=arm['source_git_commit'], analysis_git_commit=arm['source_git_commit'],
                  checkpoint_best_epoch=arm['best_epoch'], evaluation_source_git_commit='1' * 40,
                  evaluation_script_sha256=sha256(HERE / 'evaluate_test.py'), gate_sha256=plan['gate_sha256'],
                  epochs=80, threshold=.5, selection_metric='iou', samples=2113, checkpoint_sha256='a' * 64)
    result.update({k: False for k in ('race_pe_enabled', 'race_enabled', 'text_use_lora', 'bcdh_enabled', 'cdrr_enabled', 'boundary_loss_weight')})
    result['records'] = [dict(name=str(i), label_pixels=10, iou=.5, dice=2/3, precision=.75, recall=.6, brier=.05) for i in range(2113)]
    result.update({'macro_' + m: result['records'][0][m] for m in ('iou', 'dice', 'precision', 'recall', 'brier')})
    validate = lambda value: validate_result(value, arm, plan, '1' * 40, sha256(HERE / 'evaluate_test.py'))
    validate(result)
    bad_cases = dict(seed=2027, checkpoint_best_epoch=79, checkpoint_git_commit='0' * 40,
                     evaluation_source_git_commit='0' * 40, threshold=.51, samples=2112,
                     threshold_selected_on_test=True, macro_iou=.6, race_pe_enabled=True)
    for key, value in bad_cases.items():
        bad = copy.deepcopy(result)
        bad[key] = value
        must_reject(lambda: validate(bad))
    bad = copy.deepcopy(result)
    bad['records'][1]['name'] = '0'
    must_reject(lambda: validate(bad))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        gate = json.loads((HERE / 'three_seed_summary.json').read_text())
        gate['passed'] = False
        write_json(path / 'three_seed_summary.json', gate)
        changed_plan = copy.deepcopy(plan)
        changed_plan['gate_sha256'] = sha256(path / 'three_seed_summary.json')
        write_json(path / 'test_plan.json', changed_plan)
        must_reject(lambda: load_plan(path))
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    proof = dict(historical_evaluator_sha256=sha256(original),
                 evaluator_sha256=sha256(HERE / 'evaluate_test.py'), unchanged_scientific_functions=list(scientific),
                 inference_loop_ast_identical=True, protocol_rejections_checked=len(bad_cases) + 2,
                 valid_mock_result_accepted=True, test_inference_performed=False)
    write_json(HERE / 'evaluator_reuse_verification.json', proof)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
