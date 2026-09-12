"""Final local audit of terminal status, Git bytes, source tag and HF receipt."""
import json
from pathlib import Path
import subprocess
from analysis import digest
HERE=Path(__file__).resolve().parent;DOCS=HERE.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
runtime=read(HERE/'results/runtime.json');state=read(HERE/'state.json')
summary=read(HERE/'summary.json');backup=read(HERE/'hf_upload_verified.json')
sha=runtime['evaluation_source_git_commit']
assert runtime['phase']==state['phase']=='complete' and 1<=state['inspections']<=2
assert state['completion_observed_within30min']
assert summary['verified'] and summary['samples']==2113 and len(summary['heads'])==21
assert backup['verified'] and backup['all_sizes_and_xet_hashes_match']
assert sha==summary['evaluation_source_git_commit']==backup['evaluation_source_git_commit']
assert subprocess.check_output(['git','rev-parse','test-local-refinement-f-v2-20260912'],cwd=DOCS,text=True).strip()==sha
for name,meta in runtime['artifacts'].items():
    path=HERE/'results'/name
    assert digest(path)==meta['sha256'] and path.stat().st_size==meta['bytes']
    assert subprocess.check_output(['git','show','HEAD:'+path.relative_to(DOCS).as_posix()],cwd=DOCS)==path.read_bytes()
for entry in read(HERE/'hf_manifest.json')['files']:
    assert digest(Path(entry['local']))==entry['sha256']
print(json.dumps(dict(verified=True,evaluation_source_git_commit=sha,raw_artifacts=len(runtime['artifacts']),
    models_including_baseline=21,samples=2113,inspections=state['inspections'],
    completion_delay_seconds=state['completion_delay_seconds'],hf_files=backup['files_verified'],
    report_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip())))
