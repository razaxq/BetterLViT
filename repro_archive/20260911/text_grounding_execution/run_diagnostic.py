"""One frozen R2 Train preflight followed by the registered full Val interventions."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
MANIFEST=json.loads((HERE/'manifest.json').read_text())
ROOT=Path(MANIFEST['baseline_repository'])
os.environ.update(BETTERLVIT_EXPERIMENT='r2_single_cosine',TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',
    HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
    PYTHONHASHSEED='1219',TOKENIZERS_PARALLELISM='false',BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16')


def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        parts=[x.lower() for x in Path(os.fsdecode(args[0])).parts]
        if 'test_folder' in parts or 'test_text.xlsx' in parts:
            raise RuntimeError('Test access prohibited in phase A')


sys.addaudithook(guard)
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
os.chdir(ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import Config as config
from Load_Dataset import ImageToImage2D,ValGenerator,_build_tokenizer
from export_validation_metrics import build_model,validation_loader
from nets.LViT import LViT
from utils import read_text
sys.path.insert(0,str(HERE))
from analysis import digest,metrics,summarize,write_json
from text_policy import VARIANTS,SCOPES,TextPolicy,parse,group
from routing import separate_forward


def state_hash(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def deterministic():
    torch.manual_seed(1219);np.random.seed(1219);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


@torch.inference_mode()
def batch_diagnostic(model,tokenizer,policy,texts,frequency,batch,names,verify_native=False):
    image=batch['image'].cuda();ids=batch['input_ids'].cuda();mask=batch['attention_mask'].cuda()
    original=model.encode_text(ids,mask)
    original_p=LViT.forward(model,image,original,text_mask=mask)
    if verify_native:
        assert torch.equal(original_p,model(image,ids,mask)),'Native R2 forward differs'
        same=separate_forward(LViT.forward,model,image,original,mask,original)
        assert torch.equal(same,original_p),'Separated identity differs'
    p0=original_p[:,0].cpu().numpy();labels=batch['label'].numpy()
    rows=[]; transformed=[policy.variants(n,texts[n]) for n in names]
    for i,name in enumerate(names):
        rows.append(dict(name=name,source_text=texts[name],variant_texts=transformed[i],
            group=group(parse(texts[name])),train_template_frequency=frequency[texts[name]],
            baseline=metrics(p0[i],labels[i]),conditions={}))
    for variant in VARIANTS:
        vs=[x[variant] for x in transformed]
        # Detect accidental truncation before making fixed-length model input.
        lengths=[len(x) for x in tokenizer(vs,truncation=False)['input_ids']]
        assert max(lengths)<=MANIFEST['tokenizer_max_length'],(variant,max(lengths))
        encoded=tokenizer(vs,max_length=32,padding='max_length',truncation=True,return_tensors='pt')
        alt_ids=encoded['input_ids'].cuda();alt_mask=encoded['attention_mask'].cuda()
        changed=((ids!=alt_ids).any(1)|(mask!=alt_mask).any(1)).cpu().tolist()
        if not any(changed):alt_text=original
        else:alt_text=model.encode_text(alt_ids,alt_mask)
        for scope in SCOPES:
            main_text=alt_text if scope in ('main','both') else original
            main_mask=alt_mask if scope in ('main','both') else mask
            ep_text=alt_text if scope in ('eppa','both') else original
            prob=separate_forward(LViT.forward,model,image,main_text,main_mask,ep_text)
            if verify_native and variant=='generic' and scope=='both':
                assert torch.equal(prob,model(image,alt_ids,alt_mask)),'Both-path intervention differs from native alternate text'
                repeat=separate_forward(LViT.forward,model,image,main_text,main_mask,ep_text)
                assert torch.equal(prob,repeat),'Intervention is not deterministic'
            output=prob[:,0].cpu().numpy()
            for i,name in enumerate(names):
                item=metrics(output[i],labels[i]);delta=output[i]-p0[i]
                item.update(tokens_changed=bool(changed[i]),raw_text_changed=vs[i]!=texts[name],
                    probability_mae=float(np.abs(delta).mean(dtype=np.float64)),
                    probability_max_difference=float(np.abs(delta).max()),
                    changed_pixels=int(((output[i]>.5)!=(p0[i]>.5)).sum()))
                # Unchanged token sequences must give unchanged outputs regardless of controls.
                if not changed[i]:assert item['probability_max_difference']<1e-6,(name,variant,scope,item['probability_max_difference'])
                rows[i]['conditions'][variant+'/'+scope]=item
    return rows


def main(args,runtime):
    started=time.time();deterministic()
    def preflight_timeout(signum,frame):raise TimeoutError('Train preflight exceeded 600 seconds')
    signal.signal(signal.SIGALRM,preflight_timeout);signal.alarm(600)
    assert not MANIFEST['test_split_allowed'] and not MANIFEST['model_updates']
    assert tuple(MANIFEST['variants'])==VARIANTS and tuple(MANIFEST['scopes'])==SCOPES
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==MANIFEST['baseline_source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    assert not config.text_use_lora and config.boundary_loss_weight==0
    assert digest(HERE/'r2_validation.json')==MANIFEST['reference_validation_sha256']
    workbook=Path(config.task_dataset)/'Train_Val_text.xlsx'
    assert digest(workbook)==MANIFEST['train_workbook_sha256']
    checkpoint_hash=digest(MANIFEST['baseline_checkpoint'])
    assert checkpoint_hash==MANIFEST['baseline_checkpoint_sha256']
    checkpoint=torch.load(MANIFEST['baseline_checkpoint'],map_location='cpu',weights_only=True)
    assert checkpoint['source_git_commit']==MANIFEST['baseline_source_git_commit']
    assert checkpoint['epoch']==MANIFEST['baseline_checkpoint_epoch_zero_based']
    assert checkpoint['best_epoch']==MANIFEST['baseline_best_epoch']
    model=build_model();model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    model.requires_grad_(False);model.cuda().eval();before=state_hash(model)
    assert all(hasattr(getattr(model,n),'eppa') for n in ('up4','up3','up2','up1'))
    texts=read_text(str(workbook));tokenizer=_build_tokenizer()
    train=ImageToImage2D(config.train_dataset,config.task_name,texts,ValGenerator([224,224]),image_size=224)
    assert len(train)==MANIFEST['train_count'] and Path(train.dataset_path).resolve().name=='Train_Folder'
    membership=hashlib.sha256(('\n'.join(train.mask_list)+'\n').encode()).hexdigest()
    assert membership==MANIFEST['train_membership_sha256']
    train_texts=[texts[n] for n in train.mask_list];policy=TextPolicy(train_texts);frequency=Counter(train_texts)
    ordered=sorted(range(len(train)),key=lambda i:hashlib.sha256(('text-path-a-v1:'+train.mask_list[i]).encode()).hexdigest())
    selected=ordered[:MANIFEST['train_preflight_samples']]
    preflight_loader=DataLoader(Subset(train,selected),batch_size=16,shuffle=False,num_workers=0)
    preflight=[];timings=[]
    for batch,names in preflight_loader:
        torch.cuda.synchronize();t=time.time()
        preflight.extend(batch_diagnostic(model,tokenizer,policy,texts,frequency,batch,names,verify_native=True))
        torch.cuda.synchronize();timings.append(time.time()-t)
    assert state_hash(model)==before and all(p.grad is None for p in model.parameters())
    # Startup already occurred; extrapolate only the remaining validation work.
    eta=time.time()+max(timings)*math.ceil(MANIFEST['validation_count']/16)*1.15+120
    write_json(args.output/'preflight.json',dict(status='ok',train_names=[r['name'] for r in preflight],
        native_original_and_alternate_exact=True,separated_identity_exact=True,deterministic_repeat_exact=True,
        baseline_state_unchanged=True,train_batch_seconds=timings,expected_completion_unix=eta,
        train_membership_sha256=membership,train_workbook_sha256=digest(workbook),
        policy_semantic_groups=len(policy.pool),parser_train_coverage=sum(parse(t) is not None for t in train_texts)))
    write_json(args.output/'train_records.json',dict(split='train_preflight',records=preflight))
    runtime.update(phase='validation_running',expected_completion_unix=eta,preflight_finished_unix=time.time())
    write_json(args.output/'runtime.json',runtime)
    signal.alarm(0)
    dataset,loader=validation_loader(16)
    assert len(dataset)==MANIFEST['validation_count'] and Path(dataset.dataset_path).resolve().name=='Val_Folder'
    reference=json.loads((HERE/'r2_validation.json').read_text());refs={r['name']:r for r in reference['records']}
    assert set(refs)==set(dataset.mask_list)
    max_difference=0.;rows=[]
    for batch,names in loader:
        new=batch_diagnostic(model,tokenizer,policy,texts,frequency,batch,names)
        for row in new:
            for metric in ('iou','dice','precision','recall'):
                d=abs(row['baseline'][metric]-refs[row['name']][metric]);max_difference=max(max_difference,d)
                assert d<1e-12,(row['name'],metric,d)
        rows.extend(new)
        print(json.dumps(dict(validation_done=len(rows),validation_total=len(dataset))),flush=True)
    assert state_hash(model)==before and all(p.grad is None for p in model.parameters())
    assert digest(MANIFEST['baseline_checkpoint'])==checkpoint_hash
    metadata=dict(split='validation',test_split_accessed=False,model_updated=False,
        diagnostic_source_git_commit=args.source_sha,baseline_source_git_commit=MANIFEST['baseline_source_git_commit'],
        baseline_checkpoint_sha256=checkpoint_hash,baseline_state_sha256=before,baseline_state_unchanged=True,
        baseline_per_image_max_difference=max_difference,baseline_best_epoch=67)
    write_json(args.output/'records.json',dict(**metadata,records=rows))
    write_json(args.output/'summary.json',summarize(rows))
    write_json(args.output/'provenance.json',dict(**metadata,torch_version=torch.__version__,numpy_version=np.__version__,
        cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(),elapsed_seconds=time.time()-started,
        scripts_sha256={p.name:digest(p) for p in HERE.iterdir() if p.suffix in ('.py','.json') and p.name!='deployment.json'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True)
    args=p.parse_args();assert len(args.source_sha)==40 and all(c in '0123456789abcdef' for c in args.source_sha)
    args.output.mkdir(exist_ok=False)
    runtime=dict(phase='preflight_running',submitted_unix=time.time(),diagnostic_source_git_commit=args.source_sha,test_split_accessed=False)
    write_json(args.output/'runtime.json',runtime)
    try:
        main(args,runtime)
        runtime.update(phase='complete',completed_unix=time.time(),artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.is_file() and p.name!='runtime.json'})
    except BaseException:
        runtime.update(phase='failed',completed_unix=time.time(),traceback=traceback.format_exc())
        raise
    finally:write_json(args.output/'runtime.json',runtime)
