"""Publish only this experiment's documents and tracker to the actual GitHub remote."""
import argparse,json,subprocess
from remote_ops import HERE,DOCS,read,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--message',required=True);p.add_argument('--completed-label',choices=('rs1s2027','rs1s3407'));a=p.parse_args()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=DOCS,text=True).strip()=='docs/experiment-tracker'
    if a.completed_label:
        folder=HERE/(a.completed_label+'_results')
        for filename in ('independent_verification.json','hf_upload_verified.json','download_xet_verified.json'):
            assert read(folder/filename)['verified']
    allowed=[str(HERE.relative_to(DOCS)).replace('\\','/'),'docs/EXPERIMENT_TRACKER.md']
    already=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=DOCS,text=True).splitlines()
    assert all(n==allowed[1] or n.startswith(allowed[0]+'/') for n in already),'Unrelated staged changes'
    subprocess.run(['git','add','--',*allowed],cwd=DOCS,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=DOCS).returncode:
        subprocess.run(['git','commit','-m',a.message],cwd=DOCS,check=True,capture_output=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
    github='https://github.com/razaxq/BetterLViT.git';branch='refs/heads/docs/experiment-tracker'
    subprocess.run(['git','push',github,'HEAD:'+branch],cwd=DOCS,check=True,capture_output=True)
    actual=subprocess.check_output(['git','ls-remote',github,branch],cwd=DOCS,text=True).split()[0];assert actual==sha
    proof=dict(verified=True,documentation_commit=sha,github=github,ref=branch)
    name=(a.completed_label+'_results/github_archive_verified.json') if a.completed_label else 'github_execution_verified.json'
    save(name,proof)
    print(json.dumps(proof))

if __name__=='__main__':main()
