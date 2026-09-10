"""Fetch immutable completed-run evidence; refuses live training snapshots."""
import argparse
import json
from pathlib import Path
import subprocess
from remote_ops import HERE,KEY,remote,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('rs1','rs2','rs3'),required=True);label=p.parse_args().label
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
    assert ck['source_git_commit']==SOURCE['source_git_commit'] and ck['seed']==1219 and ck['epochs']==80
    assert ck['regional_supervision']['mode']==runtime['manifest']['regional_mode']
    assert ck['regional_supervision']['weight']==runtime['manifest']['regional_weight']
    row=dict(role=role,path=str(path),bytes=path.stat().st_size,source_git_commit=ck['source_git_commit'],
        epoch=ck['epoch'],best_epoch=ck['best_epoch'],regional_supervision=ck['regional_supervision'],training_recipe=ck['training_recipe'])
    models.append(row)
    if role=='last':
        assert len(ck['epoch_history'])==80
        (run/'epoch_history.json').write_text(json.dumps(ck['epoch_history'],indent=2)+'\\n')
    del ck
for epoch in (20,40,60,80):
    d=json.loads((run/'diagnostics'/f'epoch_{epoch:03d}.json').read_text())
    assert d['state_sha256_before']==d['state_sha256_after'] and d['gradient_state_unchanged'] and d['rng_restored']
(run/'checkpoint_metadata.json').write_text(json.dumps(models,indent=2)+'\\n')
print(json.dumps(dict(source=SOURCE,models=models,validation={k:v for k,v in val.items() if k!='records'},runtime=runtime)))
''',timeout=180)
    destination=HERE/(label+'_results');destination.mkdir(exist_ok=True)
    subprocess.run(['scp','-r','-i',KEY,'-P','21465','-o','BatchMode=yes',
        'root@connect.westb.seetacloud.com:'+source['remote_run']+'/.',str(destination)],check=True)
    save(label+'_results/verified.json',value)
    script=Path(source['local_repository'])/'tools/compare_regional.py'
    baseline=HERE.parents[1]/'20260908/recipe_execution/r2_results/validation.json'
    subprocess.run(['python','-X','utf8',str(script),'--control',str(baseline),'--candidate',str(destination/'validation.json'),
        '--output',str(destination/('r2_vs_'+label+'.json'))],check=True)
    if label in ('rs2','rs3'):
        subprocess.run(['python','-X','utf8',str(script),'--control',str(HERE/'rs1_results/validation.json'),
            '--candidate',str(destination/'validation.json'),'--output',str(destination/('rs1_vs_'+label+'.json')),'--regional-increment'],check=True)
    d=value['validation'];gate=json.loads((destination/('r2_vs_'+label+'.json')).read_text())
    body=f"# {label.upper()}完成记录\n\n来源 `{source['source_git_commit']}`；80轮，Best {d['checkpoint_best_epoch']}。\n\n"
    body+=f"Val macro IoU {100*d['macro_iou']:.4f}%，Dice {100*d['macro_dice']:.4f}%；R2筛选门通过：{gate['passed']}。\n\n"
    body+="本阶段未访问Test。四次Train遥测模型/梯度状态不变，完整历史、检查点来源与Val结果已核验；HF备份状态需另记录，不能仅凭本地归档宣称已上传。\n"
    (destination/'README.md').write_text(body,encoding='utf-8')
    print(json.dumps(dict(label=label,archived=str(destination),passed=gate['passed'])))


if __name__=='__main__':main()
