"""Five Train-only steps: historical control parity and audit noninterference."""
import argparse,json,subprocess
from remote_ops import HERE,DOCS,read,remote,save,environment

def execute(label):
    s=read(HERE/'sources.json')[label]
    for name,args in [('disabled',['--disable-regional']),('first',[]),('audit',['--audit-dir',s['remote_run']+'_preflight_audit'])]:
        value=remote('SOURCE='+repr(s)+'\nENV='+repr(environment(s))+'\nARGS='+repr(args)+'\n'+'''
import json,os,subprocess
p=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','tools/preflight_regional.py',*ARGS],cwd=SOURCE['repository'],
    env=dict(os.environ,**ENV),capture_output=True,text=True,timeout=210)
result=next((json.loads(line) for line in reversed(p.stdout.splitlines()) if line.startswith('{')),None)
print(json.dumps(dict(returncode=p.returncode,result=result,stdout=p.stdout,stderr=p.stderr)))
''',timeout=240)
        save('preflight/'+label+'_'+name+'_execution.json',value)
        assert value['returncode']==0,(name,value['stderr'][-4500:])
        assert value['result']['status']=='ok'
        save('preflight/'+label+'_'+name+'.json',value['result'])
        print(json.dumps(dict(label=label,phase=name,status='ok',seconds=value['result']['seconds_each_step'])),flush=True)

def verify():
    sources=read(HERE/'sources.json');github=read(HERE/'github_sources_verified.json');candidates={}
    assert github['verified']
    for label,s in sources.items():
        for ref in ('refs/heads/'+s['branch'],'refs/tags/'+s['experiment_tag']):assert github['refs'][ref]==s['source_git_commit']
        changed=subprocess.check_output(['git','diff','--name-only',s['replication_parent'],s['source_git_commit']],cwd=s['local_repository'],text=True).splitlines()
        assert sorted(changed)==sorted(s['changed_files_from_rs1'])
        baseline=read(DOCS/('repro_archive/20260909/recipe_replication/r2s'+str(s['seed'])+'_preflight_1.json'))
        first=read(HERE/('preflight/'+label+'_first.json'));audit=read(HERE/('preflight/'+label+'_audit.json'));disabled=read(HERE/('preflight/'+label+'_disabled.json'))
        legacy=read(HERE/('preflight/'+label+'_legacy_disabled.json'));grouped=read(HERE/('preflight/'+label+'_formal_r2_grouped.json'))
        assert grouped['source_git_commit']==s['baseline']['source_git_commit'] and grouped['seed']==s['seed']
        assert legacy['source_git_commit']==s['source_git_commit'] and legacy['seed']==s['seed']
        for d in (legacy,grouped):assert d['status']=='ok' and not d['test_split_accessed'] and not d['formal_training_performed']
        for d in (first,audit,disabled):
            assert d['source_git_commit']==s['source_git_commit'] and d['seed']==s['seed']
            assert d['status']=='ok' and not d['formal_training_performed'] and not d['test_split_accessed']
            for k in ('initial_base_sha256','input_image_sha256','first_output_sha256'):assert d[k]==baseline[k],(label,k)
            assert d['initial_rng']==first['initial_rng']
        for k in ('output_sha256_each_step','loss_each_step'):
            assert legacy[k]==baseline[k],(label,'legacy historical smoke-test five-step parity',k)
            assert disabled[k]==grouped[k],(label,'actual formal R2 grouped optimizer five-step parity',k)
            assert first[k]==audit[k],(label,'audit noninterference',k)
        assert first['weight']==.128312 and not first['audit_exercised'] and audit['audit_exercised'] and disabled['regional_disabled']
        candidates[label]=dict(source_git_commit=s['source_git_commit'],seed=s['seed'],legacy_r2_smoke_test_five_steps_exact=True,
            actual_formal_r2_grouped_optimizer_five_steps_exact=True,
            audit_repeat_exact=True,loss_and_model_unchanged_from_rs1=True,peak_allocated_bytes=audit['peak_allocated_bytes'],
            steady_seconds_per_batch=first['steady_seconds_per_batch'])
    value=dict(status='ok',candidates=candidates,test_split_accessed=False,formal_training_performed=False)
    save('preflight_verified.json',value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('rs1s2027','rs1s3407'));p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:print(json.dumps(verify(),indent=2))
    else:assert a.label;execute(a.label)
