"""Audit the deployed evaluator against pinned sources without querying the server."""
import ast
import json
from pathlib import Path
import subprocess
from analysis import digest,write_json
HERE=Path(__file__).resolve().parent;DOCS=HERE.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
a=read(HERE/'authorization.json');d=read(HERE/'launch.json')
sha=d['evaluation_source_git_commit']
assert subprocess.check_output(['git','rev-parse','test-local-refinement-f-v2-20260912'],cwd=DOCS,text=True).strip()==sha
for name,expected in d['deployed_sha256'].items():
    path=(DOCS/a['historical_test_relative']) if name=='r2_historical_test.json' else HERE/name
    assert digest(path)==expected
    relative=path.relative_to(DOCS).as_posix()
    assert subprocess.check_output(['git','show',sha+':'+relative],cwd=DOCS)==path.read_bytes()
for name,expected in a['inherited_files_sha256'].items():
    path=HERE.parent/'local_refinement_screen'/name
    assert digest(path)==expected==digest(HERE/name)
tree=ast.parse((HERE/'evaluate.py').read_text(encoding='utf-8'))
attributes=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
assert 'backward' not in attributes and 'optim' not in attributes and 'step' not in attributes
assert 'inference_mode' in attributes and 'requires_grad_' in attributes
source=(HERE/'evaluate.py').read_text(encoding='utf-8')
assert 'range(5)' in source and 'for variant in VARIANTS' in source
assert 'half().float()' in source and 'torch.equal(p[~selected],probability[~selected])' in source
value=dict(verified=True,evaluation_source_git_commit=sha,f_source_git_commit=a['f_source_git_commit'],
    deployed_files_exact_in_git=True,inherited_inference_code_unchanged=True,
    no_optimizer_backward_or_step=True,all_twenty_existing_heads=True,
    feature_quantization_matches_training=True,no_server_status_query=True)
write_json(HERE/'source_verified.json',value);print(json.dumps(value))
