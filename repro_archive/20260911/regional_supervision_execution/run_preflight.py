"""One bounded preflight; persists full logs and structured output."""
import argparse,json
from remote_ops import HERE,remote,save,environment


def execute(label,name,profile,args):
    source=json.loads((HERE/(label+'_deployment.json')).read_text())
    repo=source['repository'];env=environment(source['source_git_commit'],profile)
    script='tools/check_regional_objective.py' if name=='checks' else 'tools/preflight_regional.py'
    value=remote('REPO='+repr(repo)+'\nENV='+repr(env)+'\nARGS='+repr(args)+'\nSCRIPT='+repr(script)+'\n'+'''
import json,os,subprocess
p=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python',SCRIPT]+ARGS,cwd=REPO,
    env=dict(os.environ,**ENV),capture_output=True,text=True,timeout=160)
lines=p.stdout.splitlines()
result=next((json.loads(line) for line in reversed(lines) if line.startswith('{')),None)
print(json.dumps(dict(returncode=p.returncode,result=result,stdout=p.stdout,stderr=p.stderr)))
''',timeout=180)
    save('preflight/'+name+'_execution.json',value)
    assert value['returncode']==0,(name,value['stderr'][-4500:])
    assert value['result'] and value['result']['status']=='ok'
    save('preflight/'+name+'.json',value['result'])
    return {k:v for k,v in value['result'].items() if k!='batches'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--name',required=True)
    p.add_argument('--profile',required=True);args,extra=p.parse_known_args()
    print(json.dumps(execute(args.label,args.name,args.profile,extra),indent=2))
