"""Prepare or upload only an already-confirmed completed recipe's static files."""
import argparse
import json
import subprocess
from pathlib import Path
from prepare_recipe import ROOT, HERE, KEY, remote

p=argparse.ArgumentParser()
p.add_argument('action',choices=('prepare','upload'))
p.add_argument('--label',choices=('r1','r2'),required=True)
args=p.parse_args()
source=json.loads((HERE/'sources.json').read_text())[args.label]
snapshot=json.loads((ROOT/f'{args.label}_final_snapshot.json').read_text(encoding='utf-8'))
assert snapshot['files']['runtime.json']['phase']=='complete'
assert snapshot['source_git_commit']==source['source_git_commit']
if args.action=='prepare':
    value=remote('SOURCE='+repr(source)+'\n'+'''
import json,subprocess
command=['/root/autodl-tmp/envs/betterlvit-paper/bin/python','/root/recipe_runs/prepare_hf_recipe_server.py',
 '--run',SOURCE['remote_run'],'--source',SOURCE['source_git_commit']]
result=subprocess.run(command,capture_output=True,text=True)
assert result.returncode==0,result.stderr+result.stdout[-3000:]
print(json.dumps(json.loads(result.stdout.splitlines()[-1])))
''',timeout=180)
    (ROOT/f'{args.label}_hf_manifest.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))
else:
    from huggingface_hub import get_token
    token=get_token()
    assert token
    command='PYTHONPATH=/root/autodl-tmp/hf-bucket-client /root/autodl-tmp/envs/betterlvit-paper/bin/python /root/recipe_runs/upload_hf_recipe_server.py --manifest /root/recipe_runs/hf_staging/'+source['source_git_commit'][:8]+'_manifest.json'
    # Credential is scoped to the authorized HF request via stdin, never logged or written.
    result=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'root@connect.westb.seetacloud.com',command],input=token+'\n',text=True,capture_output=True,timeout=600)
    (ROOT/f'{args.label}_hf_upload.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr[-2500:]+result.stdout[-2500:])
    value=json.loads(result.stdout.splitlines()[-1])
    assert value['verified'] and value['verified_files']==5
    (ROOT/f'{args.label}_hf_upload_verified.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))
