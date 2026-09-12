"""Run CPU regressions and one bounded actual Train preflight per invocation."""
import argparse
import json
from remote_ops import HERE,read,remote,save,environment
ap=argparse.ArgumentParser()
ap.add_argument('mode',choices=('cpu','baseline','disabled','t1','t2','t1_observe','t2_observe','verify'))
mode=ap.parse_args().mode
sources=read(HERE/'sources.json')
if mode=='verify':
    base=read(HERE/'preflight/baseline.json');disabled=read(HERE/'preflight/disabled.json')
    for k in ('initial_base_sha256','input_image_sha256','input_text_sha256','initial_rng','output_sha256_each_step','loss_each_step'):
        assert base[k]==disabled[k],('baseline parity',k)
    candidates={}
    for label in ('t1','t2'):
        v=read(HERE/('preflight/'+label+'.json'));o=read(HERE/('preflight/'+label+'_observe.json'))
        assert v['source_git_commit']==sources[label]['source_git_commit']
        for k in ('initial_base_sha256','input_image_sha256','input_text_sha256','initial_rng'):
            assert v[k]==base[k],(label,k)
        assert v['output_sha256_each_step'][0]==base['output_sha256_each_step'][0]
        for k in ('output_sha256_each_step','loss_each_step','gradient_absmax_each_step'):
            assert v[k]==o[k],(label,'observation parity',k)
        assert v['adapter_parameters']==16896 and v['status']=='ok'
        assert not v['test_split_accessed'] and not v['val_split_accessed']
        candidates[label]=dict(source_git_commit=v['source_git_commit'],peak_allocated_bytes=v['peak_allocated_bytes'],
            steady_seconds_per_batch=v['steady_seconds_per_batch'],shapes=v['shapes'],
            residual_rms_each_step=v['residual_rms_each_step'])
    assert read(HERE/'preflight/t1.json')['initial_adapter_sha256']==read(HERE/'preflight/t2.json')['initial_adapter_sha256']
    assert read(HERE/'preflight/cpu.json')['status']=='ok'
    proof=dict(verified=True,baseline_five_steps_exact=True,initial_weights_rng_and_output_exact=True,
        observations_noninterference_exact=True,matched_parameters=16896,candidates=candidates,
        formal_training_performed=False,test_split_accessed=False,val_split_accessed=False)
    save('preflight_verified.json',proof);print(json.dumps(proof,indent=2))
else:
    label='t2' if mode.startswith('t2') else 't1'
    s=sources[label];env=environment(s)
    args=[]
    if mode=='baseline':
        args=['--baseline-root','/root/BetterLViT-recipe-r2']
        env['BETTERLVIT_EXPERIMENT']='r2_single_cosine'
        env['BETTERLVIT_GIT_COMMIT']='9eca26de5b301099805530edbf5a1a8718bea662'
    if mode=='disabled':args=['--disable-adapter']
    if mode.endswith('_observe'):args=['--observe']
    script='tools/check_decoder_context.py' if mode=='cpu' else 'tools/preflight_decoder_context.py'
    value=remote('SOURCE='+repr(s)+'\nENV='+repr(env)+'\nARGS='+repr(args)+'\nSCRIPT='+repr(script)+'\n'+'''
import json,os,subprocess
cwd='/root/BetterLViT-recipe-r2' if '--baseline-root' in ARGS else SOURCE['repository']
script=SOURCE['repository']+'/'+SCRIPT
p=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python',script,*ARGS],cwd=cwd,
    env=dict(os.environ,**ENV),capture_output=True,text=True,timeout=240)
result=next((json.loads(line) for line in reversed(p.stdout.splitlines()) if line.startswith('{')),None)
print(json.dumps(dict(returncode=p.returncode,result=result,stdout=p.stdout,stderr=p.stderr)))
''',timeout=270)
    save('preflight/'+mode+'_execution.json',value)
    assert value['returncode']==0,value['stderr'][-5500:]
    save('preflight/'+mode+'.json',value['result'])
    print(json.dumps(dict(mode=mode,result=value['result']),indent=2))
