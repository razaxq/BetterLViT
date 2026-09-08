"""Apply the registered C4 gate after checking the recipe's frozen manifest."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_recipe import validate_recipe_manifest
from compare_visual_validation import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    control, candidate, manifest = [json.loads(p.read_text()) for p in
                                    (args.control, args.candidate, args.manifest)]
    validate_recipe_manifest(manifest)
    assert candidate['experiment'] == manifest['profile']
    assert candidate['seed'] == manifest['seed']
    recipe = candidate['training_recipe']
    for key in ('augmentation_policy', 'lr_schedule', 'planned_epoch_lrs', 'epochs'):
        assert recipe[key] == manifest[key], key
    assert recipe['optimizer'] == 'Adam' and recipe['weight_decay'] == 1e-4
    assert control['checkpoint_git_commit'] == manifest['control_source_git_commit']
    result = compare(control, candidate, role='c4')
    result['training_recipe'] = recipe
    result['source_json_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.control, args.candidate, args.manifest)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
