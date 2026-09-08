"""Capture installed packages and hardware once before formal training."""
import json
from prepare_recipe import remote, ROOT

value = remote('''
import importlib.metadata, json, platform, subprocess, time
import torch
print(json.dumps(dict(recorded_unix=time.time(),python=platform.python_version(),
    platform=platform.platform(),torch=str(torch.__version__),torch_cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),
    packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()],key=lambda d:d['name'].lower()),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
    dataset_root='/root/autodl-tmp/datasets/Covid19',
    dataset_contract='repro_archive/20260908/strategy_review/data_contract_audit.json')))
''')
(ROOT/'environment.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in value.items() if k!='packages'}))
