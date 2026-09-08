"""Transfer already-confirmed completed artifacts; no process/status polling."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from prepare_replication import KEY, HERE, ROOT, remote


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',choices=('c4s2027','r2s2027','c4s3407','r2s3407'),required=True)
    label=p.parse_args().label
    snap=json.loads((ROOT/f'{label}_final_snapshot.json').read_text(encoding='utf-8'))
    assert snap['files']['runtime.json']['phase']=='complete'
    source=json.loads((HERE/'sources.json').read_text())[label]
    assert snap['source_git_commit']==source['source_git_commit']
    out=HERE/(label+'_results')
    out.mkdir(exist_ok=True)
    names=['training.log','validation.log','epoch_timing.jsonl','runtime.json','validation.json']
    hashes=remote('RUN='+repr(source['remote_run'])+'\nNAMES='+repr(names)+'\n'+'''
import hashlib,json
from pathlib import Path
run=Path(RUN)
assert run.parent==Path('/root/recipe_runs')
print(json.dumps({name:dict(bytes=(run/name).stat().st_size,sha256=hashlib.sha256((run/name).read_bytes()).hexdigest()) for name in NAMES}))
''')
    for name in names:
        subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',
            'root@connect.westb.seetacloud.com:'+source['remote_run']+'/'+name,str(out/name)],check=True)
        assert (out/name).stat().st_size==hashes[name]['bytes']
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==hashes[name]['sha256']
    result=json.loads((out/'validation.json').read_text())
    assert result==snap['files']['validation.json']
    history={}
    columns=('train_loss','train_dice','train_iou','val_loss','val_dice','val_iou','lr')
    for line in (out/'training.log').read_text(encoding='utf-8').splitlines():
        parts=[x.strip() for x in line.split('|')]
        if len(parts)==9 and parts[0].isdigit():
            row=dict(zip(columns,map(float,parts[1:8])))
            assert all(math.isfinite(v) for v in row.values())
            history[int(parts[0])]=dict(epoch=int(parts[0]),**row)
    assert sorted(history)==list(range(1,81))
    planned=json.loads((ROOT/f'{label}_manifest.json').read_text())['planned_epoch_lrs']
    for epoch,row in history.items():
        # The original log uses scientific notation with two decimal places.
        assert row['lr']==float(format(planned[epoch-1],'.2e')), (epoch,row['lr'],planned[epoch-1])
    best=result['checkpoint_best_epoch']
    assert abs(history[best]['val_iou']-result['macro_iou'])<.000051
    (out/'epoch_history_rounded.json').write_text(json.dumps(list(history.values()),indent=2)+'\n',encoding='utf-8')
    for name in (f'{label}_final_snapshot.json',f'{label}_summary.json',f'c4_vs_r2_seed{source["seed"]}.json',f'{label}_state.json',f'{label}_forecast.json'):
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,out/name)
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(label=label,artifact_directory=str(out),verified_files=len(names),epoch_lr_log_checked=True)))


if __name__=='__main__':
    main()
