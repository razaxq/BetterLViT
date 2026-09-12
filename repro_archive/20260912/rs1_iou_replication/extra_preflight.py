"""Reconcile legacy smoke-test Adam with the actual grouped optimizer in formal R2."""
import argparse,ast,hashlib,json
from pathlib import Path
from remote_ops import HERE,read,remote,save,environment

LEGACY="    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)"
GROUPED="    from train_model import build_optimizer_parameter_groups\n    groups, _, _ = build_optimizer_parameter_groups(model, config.weight_decay)\n    optimizer = torch.optim.Adam(groups, lr=config.learning_rate)"

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True,choices=('rs1s2027','rs1s3407'));label=p.parse_args().label
    source=read(HERE/'sources.json')[label];baseline=source['baseline']
    def optimizer_ast(repo):
        tree=ast.parse((Path(repo)/'train_model.py').read_text(encoding='utf-8'))
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_optimizer_parameter_groups'))
    assert optimizer_ast(source['local_repository'])==optimizer_ast(baseline['local_repository'])
    for phase in ('legacy_disabled','formal_r2_grouped'):
        env=environment(source)
        if phase=='formal_r2_grouped':env.update(BETTERLVIT_EXPERIMENT='r2_single_cosine',BETTERLVIT_GIT_COMMIT=baseline['source_git_commit'])
        result=remote('SOURCE='+repr(source)+'\nENV='+repr(env)+'\nPHASE='+repr(phase)+'\nLEGACY='+repr(LEGACY)+'\nGROUPED='+repr(GROUPED)+'\n'+'''
import hashlib,json,os,subprocess
from pathlib import Path
repo=Path(SOURCE['repository'] if PHASE=='legacy_disabled' else SOURCE['baseline']['repository'])
before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert before==ENV['BETTERLVIT_GIT_COMMIT']
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
python='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
provenance={}
if PHASE=='legacy_disabled':
    command=[python,'tools/preflight_regional.py','--legacy-optimizer','--disable-regional']
else:
    script=repo/'tools/preflight_recipe.py';original=script.read_text();assert original.count(LEGACY)==1
    adapted=original.replace(LEGACY,GROUPED)
    provenance=dict(original_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        adapted_utf8_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
        adaptation='Only replace smoke-test optimizer construction with the formal trainer parameter-group builder; execute in memory; do not edit original source')
    command=[python,'-c',"exec(compile("+repr(adapted)+","+repr(str(script))+",'exec'),{'__name__':'__main__','__file__':"+repr(str(script))+"})"]
p=subprocess.run(command,cwd=repo,env=dict(os.environ,**ENV),capture_output=True,text=True,timeout=210)
result=next((json.loads(line) for line in reversed(p.stdout.splitlines()) if line.startswith('{')),None)
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
print(json.dumps(dict(returncode=p.returncode,result=result,stdout=p.stdout,stderr=p.stderr,provenance=provenance)))
''',timeout=240)
        save('preflight/'+label+'_'+phase+'_execution.json',result)
        assert result['returncode']==0,result['stderr'][-4000:]
        assert result['result']['status']=='ok';save('preflight/'+label+'_'+phase+'.json',result['result'])
        print(json.dumps(dict(label=label,phase=phase,status='ok')),flush=True)

if __name__=='__main__':main()
