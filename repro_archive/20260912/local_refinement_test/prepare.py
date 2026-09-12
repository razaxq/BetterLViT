"""Freeze existing model artifacts and exact inherited inference code before Test."""
import json
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
F=HERE.parent/'local_refinement_screen'
DOCS=HERE.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert not (HERE/'authorization.json').exists()
m=read(F/'manifest.json');runtime=read(F/'results/runtime.json');proof=read(F/'checkpoint_verified.json')
assert runtime['phase']=='complete' and proof['verified'] and proof['heads']==20
files={}
for name in ('refiner.py','mass_projection.py','analysis.py','screen_analysis.py','refiner_names.py'):
    raw=(F/name).read_bytes();(HERE/name).write_bytes(raw);files[name]=digest(HERE/name)
historical=DOCS/'repro_archive/20260910/recipe_test/results/r2s1219_test.json'
auth=dict(explicit_user_requested_test=True,user_request='试试测试集',authorization_date='2026-09-12',
    protocol_amendment='Explicit user Test request supersedes previous automatic Test gate for this fixed-checkpoint evaluation only. Original failed F screen retained.',
    f_directory='/root/local_refinement_f_a76e1005',f_source_git_commit=runtime['source_git_commit'],
    baseline=m,head_files={f'fold_{f}_heads.pt':runtime['artifacts'][f'fold_{f}_heads.pt'] for f in range(5)},
    head_state_sha256=proof['state_sha256'],inherited_files_sha256=files,
    historical_test_relative=historical.relative_to(DOCS).as_posix(),historical_test_sha256=digest(historical),
    samples=2113,folds=5,variants=m['variants'],primary_variant='fine_mass',batch_size=16,threshold=.5,
    no_new_training=True,no_test_selection=True,failed_training_screen_retained=True,
    estimated_seconds=360,first_collection_after_seconds=480,maximum_inspections=2)
(HERE/'authorization.json').write_text(json.dumps(auth,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(prepared=True,head_files=len(auth['head_files']),inherited_files=len(files))))
