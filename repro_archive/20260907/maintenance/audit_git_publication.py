"""Inspect unpublished Git blobs without printing their contents."""
import json
import re
import subprocess

REPO = 'D:/BetterLViT/tcsr_work'
def git(*args):
    return subprocess.check_output(['git', '-C', REPO, *args])
lines = git('rev-list','--objects','--branches','--tags','--not','--remotes=github-audit').decode().splitlines()
names = dict(line.split(' ',1) if ' ' in line else (line,'') for line in lines)
metadata = subprocess.run(['git','-C',REPO,'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                          input=('\n'.join(names)+'\n').encode(),stdout=subprocess.PIPE,check=True).stdout.decode().splitlines()
pattern = re.compile(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----')
findings=[]
count=0
total=0
for line in metadata:
    oid,kind,size=line.split()
    if kind!='blob': continue
    size=int(size); count+=1; total+=size
    if size>50_000_000:
        findings.append({'path':names[oid],'reason':'large blob','bytes':size})
        continue
    data=git('cat-file','blob',oid)
    if pattern.search(data):
        findings.append({'path':names[oid],'reason':'credential pattern'})
print(json.dumps({'new_blobs':count,'bytes':total,'findings':findings},ensure_ascii=False))
if findings: raise SystemExit(1)
