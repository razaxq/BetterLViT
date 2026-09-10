"""Match downloaded result bytes against the server's fresh upload-manifest Xet hashes."""
import argparse
import json
from pathlib import PurePosixPath
import hf_xet
from remote_ops import HERE,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('rs1','rs2','rs3'),required=True);label=p.parse_args().label
    folder=HERE/(label+'_results');manifest=json.loads((folder/'hf_manifest.json').read_text())
    root=manifest['bucket_prefix']+'/run/'
    pairs=[(folder/PurePosixPath(item['remote'][len(root):]),item) for item in manifest['files'] if item['remote'].startswith(root)]
    assert pairs
    hashes=hf_xet.hash_files([str(path) for path,_ in pairs]);rows=[]
    for (path,item),actual in zip(pairs,hashes):
        assert path.is_file() and path.stat().st_size==actual.file_size==item['bytes']
        assert actual.hash==item['xet_hash']
        rows.append(dict(local=str(path),remote=item['remote'],bytes=actual.file_size,xet_hash=actual.hash))
    result=dict(verified=True,source_git_commit=manifest['source_git_commit'],files=rows,verified_files=len(rows),
                logical_bytes=sum(row['bytes'] for row in rows),scope='Downloaded static run artifacts only; weights stay on server')
    save(label+'_results/download_xet_verified.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':main()
