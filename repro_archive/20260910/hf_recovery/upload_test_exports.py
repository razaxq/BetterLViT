"""Add completed fixed-protocol Test exports without overwriting historical evaluations."""
import json
import sys
from pathlib import Path

import hf_xet
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

from upload_pending import BUCKET, DOCS, OUT, ROOT, write_json

TEST_CODE = DOCS / 'repro_archive/20260910/recipe_test'
sys.path.insert(0, str(TEST_CODE))
from protocol import load_plan, sha256, validate_result


def main():
    disable_progress_bars()
    # The original five-file recipe uploads must finish before adding Test paths.
    assert json.loads((OUT / 'state.json').read_text())['phase'] == 'complete'
    tests = ROOT / 'outputs/recipe_test_20260910'
    state = json.loads((tests / 'state.json').read_text())
    assert state['phase'] == 'complete', 'Use the scheduled Test collector first'
    plan = load_plan(TEST_CODE)
    evaluation_sha = state['evaluation_source_git_commit']
    api = HfApi()
    assert api.whoami()['name'] == 'razaxq'
    mapping = []
    for arm in plan['arms']:
        result_path = tests / (arm['label'] + '_test.json')
        result = json.loads(result_path.read_text())
        validate_result(result, arm, plan, evaluation_sha, sha256(TEST_CODE / 'evaluate_test.py'))
        prefix = arm['source_git_commit'][:8] + '/test_evaluations/' + evaluation_sha[:8]
        mapping.extend([(result_path, prefix + '.json'),
                        (tests / (arm['label'] + '_test.log'), prefix + '.log')])
    mapping.extend([(tests / 'three_seed_test_summary.json', evaluation_sha[:8] + '/three_seed_test_summary.json'),
                    (TEST_CODE / 'test_plan.json', evaluation_sha[:8] + '/test_plan.json'),
                    (TEST_CODE / 'three_seed_summary.json', evaluation_sha[:8] + '/validation_gate.json')])
    hashes = hf_xet.hash_files([str(path) for path, _ in mapping])
    proof = dict(evaluation_source_git_commit=evaluation_sha, bucket=BUCKET, additive_only=True, files=[])
    for (path, destination), info in zip(mapping, hashes):
        parent = destination.rsplit('/', 1)[0]
        known = {x.path: x for x in api.list_bucket_tree(BUCKET, prefix=parent, recursive=True) if getattr(x, 'xet_hash', None)}
        if destination in known:
            assert known[destination].size == info.file_size and known[destination].xet_hash == info.hash
        else:
            api.batch_bucket_files(BUCKET, add=[(path.read_bytes(), destination)])
        known = {x.path: x for x in api.list_bucket_tree(BUCKET, prefix=parent, recursive=True) if getattr(x, 'xet_hash', None)}
        assert known[destination].size == info.file_size and known[destination].xet_hash == info.hash
        proof['files'].append(dict(source=str(path), remote=destination, bytes=info.file_size, xet_hash=info.hash))
        print(json.dumps(dict(verified=destination)), flush=True)
    proof.update(verified=True, verified_files=len(mapping))
    write_json(OUT / 'test_exports_upload_verified.json', proof)
    print(json.dumps(dict(complete=True, verified_files=len(mapping))), flush=True)


if __name__ == '__main__':
    main()
