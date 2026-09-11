"""Additively back up authorized completed visual-aux pilots; verify server and local listings."""
import argparse
import base64
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from huggingface_hub import HfApi,get_token
from remote_ops import HERE,KEY,remote,save

BUCKET='razaxq/BetterLViT'
HELPER=HERE.parents[1]/'20260910/hf_recovery/upload_verified_manifest.py'


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('rs1','rs2','rs3'),required=True);label=p.parse_args().label
    folder=HERE/(label+'_results');proof=json.loads((folder/'independent_verification.json').read_text())
    assert proof['verified'] and not proof['test_split_accessed']
    source=json.loads((HERE/'sources.json').read_text())[label];sha=source['source_git_commit'];prefix=sha[:8]
    token=get_token();assert token,'Standard HF credential cache is missing'
    api=HfApi(token=token);assert api.whoami()['name']=='razaxq'
    supplements={}
    for name in ('REPORT.md','independent_verification.json','r2_vs_'+label+'.json'):
        supplements['analysis/'+name]=base64.b64encode((folder/name).read_bytes()).decode()
    if (folder/'iou_reconciliation.json').exists():
        supplements['analysis/iou_reconciliation.json']=base64.b64encode((folder/'iou_reconciliation.json').read_bytes()).decode()
        supplements['analysis/reconcile_iou.py']=base64.b64encode((HERE/'reconcile_iou.py').read_bytes()).decode()
    if label in ('rs2','rs3'):supplements['analysis/rs1_vs_'+label+'.json']=base64.b64encode((folder/('rs1_vs_'+label+'.json')).read_bytes()).decode()
    if label=='rs3':
        supplements['analysis/rs2_vs_rs3.json']=base64.b64encode((folder/'rs2_vs_rs3.json').read_bytes()).decode()
    helper_b64=base64.b64encode(HELPER.read_bytes()).decode()
    manifest=remote('SOURCE='+repr(source)+'\nSUPPLEMENTS='+repr(supplements)+'\nHELPER_B64='+repr(helper_b64)+'\n'+'''
import base64,hashlib,json,shutil,subprocess
from pathlib import Path
import hf_xet
repo=Path(SOURCE['repository']);run=Path(SOURCE['remote_run']);sha=SOURCE['source_git_commit'];prefix=sha[:8]
runtime=json.loads((run/'runtime.json').read_text());assert runtime['phase']=='complete'
assert runtime['source_git_commit']==sha and runtime['training_rc']==runtime['validation_rc']==0
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
stage=run.parent/'hf_staging'/prefix;stage.mkdir(parents=True,exist_ok=True)
manifest_path=stage/'upload_manifest.json';helper=stage/'upload_verified_manifest.py'
helper.write_bytes(base64.b64decode(HELPER_B64))
if manifest_path.exists():
    manifest=json.loads(manifest_path.read_text())
else:
    archive=stage/'source.tar.gz'
    subprocess.run(['git','archive','--format=tar.gz','--output',str(archive),sha],cwd=repo,check=True)
    mapping=[(archive,prefix+'/source.tar.gz')]
    mapping.extend((p,prefix+'/run/'+p.relative_to(run).as_posix()) for p in sorted(run.rglob('*')) if p.is_file())
    best=Path(runtime['best_checkpoint']);session=best.parent.parent
    mapping.extend((p,prefix+'/models/'+p.name) for p in (best,best.with_name('last_model-BetterLViT.pth.tar')))
    mapping.append((repo/'experiment_manifests/active_regional.json',prefix+'/manifest.json'))
    events=list(session.rglob('events.out.tfevents.*'));assert len(events)==1
    mapping.append((events[0],prefix+'/tensorboard/'+events[0].name))
    for name,data in SUPPLEMENTS.items():
        path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(data))
        mapping.append((path,prefix+'/'+name))
    assert len({destination for _,destination in mapping})==len(mapping)
    hashes=hf_xet.hash_files([str(path) for path,_ in mapping])
    files=[dict(source=str(path),remote=dest,bytes=info.file_size,xet_hash=info.hash) for (path,dest),info in zip(mapping,hashes)]
    manifest=dict(source_git_commit=sha,bucket_prefix=prefix,classification='completed_validation_only_80e_pilot',
        test_split_accessed=False,training_and_validation_returncodes=[0,0],files=files,
        experiment_tag=SOURCE['experiment_tag'],checkpoint_checks=json.loads((run/'checkpoint_metadata.json').read_text()),
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\\n')
assert manifest['source_git_commit']==sha
hashes=hf_xet.hash_files([f['source'] for f in manifest['files']])
for item,actual in zip(manifest['files'],hashes):
    assert item['bytes']==actual.file_size and item['xet_hash']==actual.hash
print(json.dumps(manifest))
''',timeout=180)
    save(label+'_results/hf_manifest.json',manifest)
    expected={f['remote']:f for f in manifest['files']}
    known={f.path:f for f in api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
    assert set(known)<=set(expected),'Unexpected contents under this training prefix'
    for name,value in known.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
    stage='/root/autodl-tmp/regional_runs/hf_staging/'+prefix
    command=('ionice -c 3 nice -n 10 env PYTHONPATH=/root/autodl-tmp/hf-bucket-client '
             '/root/autodl-tmp/envs/betterlvit-paper/bin/python '+stage+'/upload_verified_manifest.py --manifest '+stage+'/upload_manifest.json')
    print(json.dumps(dict(event='upload_started',prefix=prefix,files=len(expected),bytes=sum(f['bytes'] for f in expected.values()))),flush=True)
    result=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'root@connect.westb.seetacloud.com',command],input=token+'\n',text=True,encoding='utf-8',capture_output=True,timeout=1200)
    safe=re.sub(r'hf_[A-Za-z0-9]{16,}','[REDACTED]',(result.stdout+'\n'+result.stderr).replace(token,'[REDACTED]'))
    (folder/'hf_upload.log').write_text(safe,encoding='utf-8')
    assert result.returncode==0,'Upload failed; see sanitized log'
    proof=json.loads(result.stdout.splitlines()[-1]);assert proof['verified'] and proof['files']==manifest['files']
    known={f.path:f for f in api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
    assert set(known)==set(expected)
    for name,value in known.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
    preservation=remote('MANIFEST='+repr(manifest)+'\n'+'''
import json,shutil,subprocess
from pathlib import Path
for item in MANIFEST['checkpoint_checks']:assert Path(item['path']).stat().st_size==item['bytes']
fs=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]);assert fs<20_000_000_000
print(json.dumps(dict(original_models_preserved=True,fs_bytes=fs,scratch_free_bytes=shutil.disk_usage('/root/autodl-tmp').free)))
''')
    proof.update(independent_local_listing_verified=True,completed_unix=time.time(),
        logical_bytes=sum(f['bytes'] for f in expected.values()),additive_only=True,remote_deletions=False,**preservation)
    save(label+'_results/hf_upload_verified.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('files','checkpoint_checks')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
