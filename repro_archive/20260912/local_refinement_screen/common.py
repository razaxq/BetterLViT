"""Immutable inputs, deterministic runtime and original-B-fit-only access."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
M=json.loads((HERE/'manifest.json').read_text())
ROOT=Path(M['baseline_repository']);B=Path(M['b_directory'])
os.environ.update(BETTERLVIT_EXPERIMENT='r2_single_cosine',TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',
    HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',
    TOKENIZERS_PARALLELISM='false',BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16')
def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        parts=[p.lower() for p in Path(os.fsdecode(args[0])).parts]
        if any(p in parts for p in ('val_folder','test_folder','val_text.xlsx','test_text.xlsx')):
            raise RuntimeError('Official Val/Test prohibited in F')
sys.addaudithook(guard)
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from analysis import digest,write_json,metrics
from utils import WeightedDiceFocal
from refiner import LocalRefiner,VARIANTS,predict

def setup():
    torch.set_num_threads(4);torch.manual_seed(M['seed']);np.random.seed(M['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Inputs:
    def __init__(self,fine=False):
        assert digest(HERE/'split.json')==M['split_sha256']
        assert digest(B/'split.json')==M['original_b_split_sha256']
        assert digest(B/'results/cache_manifest.json')==M['b_cache_manifest_sha256']
        assert digest(HERE/'mass_projection.py')==M['inherited_projection_sha256']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==M['baseline_source_git_commit']
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
        self.rows=json.loads((HERE/'split.json').read_text())['records']
        self.old=json.loads((B/'split.json').read_text())['records']
        self.by_index={r['index']:r for r in self.rows}
        assert all(self.old[i]['partition']=='fit' for i in self.by_index)
        self.manifest=json.loads((B/'results/cache_manifest.json').read_text())
        for name,meta in self.manifest['files'].items():assert digest(B/'cache'/name)==meta['sha256']
        self.arrays={k:np.memmap(B/'cache'/(k+'.bin'),dtype=dtype,mode='r',shape=tuple(shape)) for k,(shape,dtype) in self.manifest['shapes'].items()}
        if fine:
            cache=json.loads((HERE/'preflight/cache_manifest.json').read_text())
            assert cache['baseline_unchanged'] and cache['exact_identity_verified_samples']==4585
            for name,meta in cache['files'].items():assert digest(HERE/'cache'/name)==meta['sha256']
            self.fine=np.memmap(HERE/'cache/fine.bin',dtype='float16',mode='r',shape=tuple(M['feature_cache_shape']))
            self.indices=np.memmap(HERE/'cache/indices.bin',dtype='uint16',mode='r',shape=tuple(M['point_index_shape']))

    def old_batch(self,indices):
        assert all(i in self.by_index for i in indices),'Old B holdout cache access blocked'
        coarse=torch.from_numpy(np.array(self.arrays['features'][indices])).cuda().float()
        z=torch.from_numpy(np.array(self.arrays['logits'][indices])).cuda()
        labels=np.unpackbits(np.array(self.arrays['masks'][indices]),axis=1,count=224*224,bitorder='little').reshape(-1,1,224,224)
        y=torch.from_numpy(labels).cuda().float()
        return coarse,z,y

    def batch(self,indices,training_fold=None):
        if training_fold is not None:assert all(self.by_index[i]['fold']!=training_fold for i in indices),'Own held fold reached optimization'
        coarse,z,y=self.old_batch(indices)
        cr=[self.by_index[i]['cache_row'] for i in indices]
        fine=torch.from_numpy(np.array(self.fine[cr])).cuda().float()
        index=torch.from_numpy(np.array(self.indices[cr],dtype=np.int64)).cuda()
        return (coarse,fine,z,index),y

    def train_small_cutoff(self,fold):
        ix=[r['index'] for r in self.rows if r['fold']!=fold]
        sizes=np.unpackbits(np.array(self.arrays['masks'][ix]),axis=1,count=224*224,bitorder='little').sum(1)
        return float(np.quantile(sizes,.25))

def observe(heads,inputs,y,metadata):
    coarse,fine,z,index=inputs;base=z.sigmoid();b_np=base.cpu().numpy();ys=y.cpu().numpy()
    selected=torch.zeros_like(z,dtype=torch.bool).flatten(1).scatter(1,index,True).reshape_as(z)
    rows=[dict(**r,outputs={'baseline':metrics(b_np[j],ys[j])},diagnostics={}) for j,r in enumerate(metadata)]
    with torch.no_grad():
        for variant,head in heads.items():
            delta=head(*inputs,variant);p=predict(z,delta,index,variant)
            assert torch.equal(p[~selected],base[~selected]),'Noncandidate pixel changed'
            error=(p.double().sum((1,2,3))-base.double().sum((1,2,3))).abs()
            if variant.endswith('mass'):assert float(error.max())<=M['mass_error_tolerance']
            values=p.cpu().numpy()
            for j,row in enumerate(rows):
                b=b_np[j]>.5;q=values[j]>.5;t=ys[j]>0
                row['outputs'][variant]=metrics(values[j],ys[j])
                row['diagnostics'][variant]=dict(fixed_fp=int((b&~t&~q).sum()),fixed_fn=int((~b&t&q).sum()),
                    new_fp=int((~b&~t&q).sum()),new_fn=int((b&t&~q).sum()),
                    changed_pixels=int((q!=b).sum()),candidate_mean_abs_delta=float(delta[j].abs().mean()),
                    soft_mass_absolute_error=float(error[j]),noncandidate_probabilities_exact=True)
    return rows
