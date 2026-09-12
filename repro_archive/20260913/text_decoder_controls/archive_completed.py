"""Fetch immutable completed-run evidence; refuses live training snapshots."""
import argparse
import json
from pathlib import Path
import subprocess
from remote_ops import HERE,KEY,remote,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('t1','t2'),required=True);label=p.parse_args().label
    snapshots=[HERE/(label+'_final_snapshot.json'),HERE/(label+'_first_snapshot.json')]
    snapshot=next((json.loads(p.read_text()) for p in snapshots if p.exists() and json.loads(p.read_text())['runtime']['phase']=='complete'),None)
    assert snapshot is not None,'Need a completed inspection before static archival'
    source=json.loads((HERE/'sources.json').read_text())[label]
    value=remote('SOURCE='+repr(source)+'\n'+'''
import json,subprocess
from pathlib import Path
import torch
repo=Path(SOURCE['repository']);run=Path(SOURCE['remote_run'])
runtime=json.loads((run/'runtime.json').read_text());assert runtime['phase']=='complete'
assert runtime['source_git_commit']==SOURCE['source_git_commit']
val=json.loads((run/'validation.json').read_text())
assert val['checkpoint_git_commit']==val['analysis_git_commit']==SOURCE['source_git_commit']
assert val['samples']==1429 and val['threshold']==.5 and not val['test_split_accessed']
best=Path(runtime['best_checkpoint']);last=best.with_name('last_model-BetterLViT.pth.tar')
models=[]
for role,path in (('best',best),('last',last)):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    assert ck['source_git_commit']==SOURCE['source_git_commit'] and ck['seed']==SOURCE['seed'] and ck['epochs']==80
    assert ck['decoder_context']['mode']==runtime['manifest']['decoder_context_mode']
    assert ck['decoder_context']['parameters']==16896
    row=dict(role=role,path=str(path),bytes=path.stat().st_size,source_git_commit=ck['source_git_commit'],
        epoch=ck['epoch'],best_epoch=ck['best_epoch'],decoder_context=ck['decoder_context'],training_recipe=ck['training_recipe'])
    models.append(row)
    if role=='last':
        assert ck['epoch']==79 and len(ck['epoch_history'])==80
        (run/'epoch_history.json').write_text(json.dumps(ck['epoch_history'],indent=2)+'\\n')
    del ck
observations=[json.loads(line) for line in (run/'decoder_observations.jsonl').read_text().splitlines()]
assert [o['epoch'] for o in observations]==[1,10,40,80]
assert all(o['source_git_commit']==SOURCE['source_git_commit'] for o in observations)
(run/'checkpoint_metadata.json').write_text(json.dumps(models,indent=2)+'\\n')
print(json.dumps(dict(source=SOURCE,models=models,validation={k:v for k,v in val.items() if k!='records'},runtime=runtime)))
''',timeout=180)
    destination=HERE/(label+'_results');destination.mkdir(exist_ok=True)
    subprocess.run(['scp','-r','-i',KEY,'-P','21465','-o','BatchMode=yes',
        'root@connect.westb.seetacloud.com:'+source['remote_run']+'/.',str(destination)],check=True)
    save(label+'_results/verified.json',value)
    print(json.dumps(dict(label=label,archived=str(destination),source_git_commit=source['source_git_commit'])))

if __name__=='__main__':main()
