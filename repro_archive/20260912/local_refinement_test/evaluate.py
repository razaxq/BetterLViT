"""One frozen Test pass for R2 and all twenty existing F heads; no optimization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
A=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
M=A['baseline'];ROOT=Path(M['baseline_repository'])
os.environ.update(BETTERLVIT_EXPERIMENT='r2_single_cosine',TEST_SPLIT_ALLOWED='1',AUTO_TEST_EVALUATE='0',
    HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',
    TOKENIZERS_PARALLELISM='false',BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16',
    BETTERLVIT_SEED='1219',PYTHONDONTWRITEBYTECODE='1')
def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        parts=[p.lower() for p in Path(os.fsdecode(args[0])).parts]
        if any(p in parts for p in ('val_folder','train_folder','val_text.xlsx','train_val_text.xlsx')):
            raise RuntimeError('This evaluation permits only the authorized Test split')
sys.addaudithook(guard)
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from analysis import digest,metrics,write_json
from refiner import LocalRefiner,VARIANTS,predict

def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def group_id(name):
    match=re.search(r'(sub-S\d+)',name)
    return match.group(1) if match else 'file:'+name

def main(args,runtime):
    assert A['explicit_user_requested_test'] and A['no_test_selection'] and A['threshold']==.5
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==M['baseline_source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    for name,sha in A['inherited_files_sha256'].items():assert digest(HERE/name)==sha
    assert digest(HERE/'r2_historical_test.json')==A['historical_test_sha256']
    previous=json.loads((HERE/'r2_historical_test.json').read_text())
    reference={r['name']:r for r in previous['records']}
    assert len(reference)==2113 and previous['checkpoint_sha256']==M['baseline_checkpoint_sha256']
    assert digest(M['baseline_checkpoint'])==M['baseline_checkpoint_sha256']
    torch.set_num_threads(4);torch.manual_seed(1219);np.random.seed(1219)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import Config as config
    from Load_Dataset import ImageToImage2D,ValGenerator
    from export_validation_metrics import build_model
    from utils import read_text
    os.chdir(ROOT)
    assert not config.text_use_lora and config.boundary_loss_weight==0 and config.img_size==224
    torch.backends.cudnn.enabled=config.cudnn_enabled
    checkpoint=torch.load(M['baseline_checkpoint'],map_location='cpu',weights_only=True)
    assert checkpoint['source_git_commit']==M['baseline_source_git_commit'] and checkpoint['epoch']==66
    assert checkpoint['selection_metric']=='iou' and checkpoint['seed']==1219
    model=build_model();model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    model.requires_grad_(False);model.cuda().eval();before=state_hash(model)
    heads={};head_before={}
    for fold in range(5):
        name=f'fold_{fold}_heads.pt';path=Path(A['f_directory'])/'results'/name
        assert path.stat().st_size==A['head_files'][name]['bytes'] and digest(path)==A['head_files'][name]['sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['source_git_commit']==A['f_source_git_commit'] and value['fold']==fold and value['steps']==2048 and value['manifest']==M
        for variant in VARIANTS:
            key=f'{fold}/{variant}';head=LocalRefiner();head.load_state_dict(value['heads'][variant],strict=True)
            head.requires_grad_(False);head.cuda().eval()
            assert all(torch.isfinite(t).all() for t in head.state_dict().values())
            assert state_hash(head)==A['head_state_sha256'][key]
            heads[key]=head;head_before[key]=state_hash(head)
    if args.preflight_only:
        write_json(args.output/'preflight.json',dict(verified=True,evaluation_source_git_commit=args.source_sha,
            head_states=head_before,baseline_state_sha256=before,test_images_loaded=0,heads_loaded=20))
        return
    workbook=Path(config.test_dataset)/'Test_text.xlsx'
    texts=read_text(str(workbook))
    dataset=ImageToImage2D(config.test_dataset,config.task_name,texts,ValGenerator([224,224]),image_size=224)
    assert len(dataset)==2113 and Path(dataset.dataset_path).name=='Test_Folder'
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    captured={};hooks=[model.up1.register_forward_hook(lambda module,inputs,out:captured.update(feature=out)),
        model.outc.register_forward_hook(lambda module,inputs,out:captured.update(logits=out))]
    rows=[];began=time.time();runtime.update(phase='evaluating',samples=0);write_json(args.output/'runtime.json',runtime)
    with torch.inference_mode():
        for batch,names in loader:
            probability=model(batch['image'].cuda(),batch['input_ids'].cuda(),batch['attention_mask'].cuda())
            z=captured['logits'];feature=captured['feature'];assert torch.equal(probability,z.sigmoid())
            uncertainty=probability*(1-probability)
            index=torch.argsort(uncertainty.flatten(1),dim=1,descending=True,stable=True)[:,:1024]
            assert all(len(torch.unique(i))==1024 for i in index)
            coarse=F.avg_pool2d(feature,8).half().float()
            fine=feature.flatten(2).transpose(1,2).gather(1,index[:,:,None].expand(-1,-1,64)).half().float()
            inputs=(coarse,fine,z,index)
            base=probability.cpu().numpy();ys=batch['label'].numpy()[:,None]
            selected=torch.zeros_like(z,dtype=torch.bool).flatten(1).scatter(1,index,True).reshape_as(z)
            current=[]
            for j,name in enumerate(names):
                metric=metrics(base[j],ys[j]);old=reference[name]
                for k in ('iou','dice','precision','recall'):assert metric[k]==old[k],(name,k,metric[k],old[k])
                assert metric['gt_area']==old['label_pixels'] and metric['hard_area']==old['prediction_pixels']
                current.append(dict(name=name,group_id=group_id(name),outputs={'baseline':metric},diagnostics={}))
            for key,head in heads.items():
                variant=key.split('/')[1];delta=head(*inputs,variant);p=predict(z,delta,index,variant)
                assert torch.equal(p[~selected],probability[~selected])
                error=(p.double().sum((1,2,3))-probability.double().sum((1,2,3))).abs()
                if variant.endswith('mass'):assert float(error.max())<=.02
                values=p.cpu().numpy()
                for j,row in enumerate(current):
                    b=base[j]>.5;q=values[j]>.5;t=ys[j]>0
                    row['outputs'][key]=metrics(values[j],ys[j])
                    row['diagnostics'][key]=dict(fixed_fp=int((b&~t&~q).sum()),fixed_fn=int((~b&t&q).sum()),
                        new_fp=int((~b&~t&q).sum()),new_fn=int((b&t&~q).sum()),changed_pixels=int((q!=b).sum()),
                        soft_mass_absolute_error=float(error[j]),noncandidate_probabilities_exact=True)
            rows.extend(current)
            if len(rows)%256==0:
                eta=time.time()+(time.time()-began)/len(rows)*(2113-len(rows))+30
                runtime.update(samples=len(rows),expected_completion_unix=eta);write_json(args.output/'runtime.json',runtime)
                print(json.dumps(dict(samples=len(rows),elapsed_seconds=time.time()-began)),flush=True)
    for hook in hooks:hook.remove()
    assert len(rows)==2113 and {r['name'] for r in rows}==set(reference)
    assert state_hash(model)==before and all(state_hash(heads[k])==h for k,h in head_before.items())
    assert all(p.grad is None for obj in [model,*heads.values()] for p in obj.parameters())
    write_json(args.output/'records.json',dict(evaluation_source_git_commit=args.source_sha,f_source_git_commit=A['f_source_git_commit'],
        split='test',test_split_accessed=True,threshold=.5,threshold_operator='>',records=rows))
    write_json(args.output/'provenance.json',dict(evaluation_source_git_commit=args.source_sha,
        f_source_git_commit=A['f_source_git_commit'],baseline_source_git_commit=M['baseline_source_git_commit'],
        authorization_sha256=digest(HERE/'authorization.json'),baseline_checkpoint_sha256=M['baseline_checkpoint_sha256'],
        inherited_files_sha256=A['inherited_files_sha256'],head_files=A['head_files'],head_states=head_before,
        all_weights_unchanged=True,no_gradients=True,no_training=True,baseline_state_sha256=before,
        historical_baseline_exact_samples=2113,historical_exact_metrics=['iou','dice','precision','recall','label_pixels','prediction_pixels'],
        no_test_selection=True,no_ensemble=True,original_f_screen_passed=False,samples=2113,heads=20,batch_size=16,
        test_text_sha256=digest(workbook),test_membership_sha256=hashlib.sha256(('\n'.join(dataset.mask_list)+'\n').encode()).hexdigest(),
        torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.time()-began,
        maximum_allocated_gpu_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source-sha',required=True)
    ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    args.output.mkdir(exist_ok=False);runtime=dict(phase='loading',evaluation_source_git_commit=args.source_sha,started_unix=time.time())
    write_json(args.output/'runtime.json',runtime)
    try:
        main(args,runtime);runtime.update(phase='complete',samples=0 if args.preflight_only else 2113,completed_unix=time.time())
        runtime['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.name!='runtime.json'}
    except BaseException:
        runtime.update(phase='failed',failed_unix=time.time(),error=traceback.format_exc());traceback.print_exc()
    write_json(args.output/'runtime.json',runtime)
    if runtime['phase']=='failed':sys.exit(1)
