"""Verify checked-out source provenance and archive the evaluation-only difference."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent
docs=root.parents[2]
repos={a:Path('D:/BetterLViT')/p for a,p in [('c4','race_pe_c4_work'),('c8','race_pe_v2_c8_work')]}
manifest=json.loads((root/'manifest.json').read_text())
files=['Config.py','paper_experiments.py','Load_Dataset.py','train_model.py',
       'Train_one_epoch.py','nets/LViT.py','nets/BetterLViT.py','nets/race_pe.py',
       'nets/race_fuse.py','race_pe_objective.py','utils.py']
out={}
for a,repo in repos.items():
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert commit==manifest[a]['source_git_commit']
    status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert not status,status
    out[a]=dict(commit=commit,tracked_source_clean=True,files={
        name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files})
old=docs/'repro_archive/20260907/race_pe_test_20260907/evaluate_test.py'
new=docs/'repro_archive/20260907/race_pe_v2_test_20260907/evaluate_test.py'
o,n=old.read_text(),new.read_text()
diff=''.join(difflib.unified_diff(o.splitlines(True),n.splitlines(True),fromfile=old.name+' (C4)',tofile=new.name+' (C8)',n=0))
(root/'evidence/evaluation_script.diff').write_text(diff)
out['evaluation_scripts']=dict(c4_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    c8_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),diff_file='evaluation_script.diff')
subjects={}
for split,relative in [('validation','race_pe_v2_results_20260907/c4_validation.json'),
                       ('test','race_pe_v2_test_20260907/c4_test.json')]:
    records=json.loads((docs/'repro_archive/20260907'/relative).read_text())['records']
    subjects[split]={match.group(1) for row in records
        if (match:=re.match(r'sub-(S\d+)_',row['name']))}
out['filename_subject_overlap']=dict(validation_subjects=len(subjects['validation']),
    test_subjects=len(subjects['test']),overlap=len(subjects['validation']&subjects['test']),
    scope='Only subject IDs explicitly encoded in sub-S filenames; Train not audited here.')
# Case identity / label sizes and per-image metrics are audited by analyze_records.
(root/'evidence/source_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(diff)
