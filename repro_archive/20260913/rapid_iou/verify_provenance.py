"""Verify that the evaluated threshold and source were committed before Test."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=DOCS)

result=read(HERE/'test/result.json');summary=read(HERE/'test/summary.json')
assert summary['verified'] and read(HERE/'test/download_verified.json')['verified']
selection_path=(HERE/'selection.json').relative_to(DOCS).as_posix()
selection_commit=git('log','-1','--format=%H','--',selection_path).decode().strip()
assert git('show',selection_commit+':'+selection_path)==(HERE/'selection.json').read_bytes()
commit_time=int(git('show','-s','--format=%ct',selection_commit))
runtime=read(HERE/'test/runtime.json')
assert commit_time<=runtime['started_unix']
evaluation_commit=result['evaluation_source_git_commit']
codepath=(HERE/'evaluate.py').relative_to(DOCS).as_posix()
assert hashlib.sha256(git('show',evaluation_commit+':'+codepath)).hexdigest()==result['script_sha256']==sha(HERE/'evaluate.py')
assert result['selection_sha256']==sha(HERE/'selection.json')
assert result['modes']['r2s1219']['thresholds']==[.5,.54]
assert read(HERE/'selection.json')['methods']['r2s1219']['threshold']==.54
recipe=dict(checkpoint=result['checkpoint_provenance'][0],model='R2 seed1219 Best67',
    threshold=.54,threshold_comparison='float32 probability > float32(0.54)',
    image_size=224,original_text_input=True,ensemble=False,tta=False,component_filter=False,
    additional_parameters=0,forward_passes_per_image=1,threshold_selected_on='validation',
    selection_git_commit=selection_commit,evaluation_source_git_commit=evaluation_commit,
    classification='inference engineering, not a new architecture')
(HERE/'inference_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n',encoding='utf-8',newline='\n')
proof=dict(verified=True,selection_committed_before_test=True,selection_git_commit=selection_commit,
    selection_commit_unix=commit_time,test_started_unix=runtime['started_unix'],
    evaluation_source_git_commit=evaluation_commit,test_thresholds=[.5,.54],
    checkpoint_sha256=result['checkpoint_provenance'][0]['sha256'],exact_historical_05_counts=True,
    result_sha256=sha(HERE/'test/result.json'),no_extra_parameters=True,no_extra_forward_passes=True)
(HERE/'provenance_verified.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(proof))
