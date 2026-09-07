import json
from pathlib import Path
import re

paths=sorted(list(Path('/root/.cache/huggingface/xet/logs').glob('*.log'))+
             list(Path('/root/maintenance_20260907/xet_serial_cache/logs').glob('*.log'))+
             list(Path('/root/maintenance_20260907/xet_clean_cache/logs').glob('*.log')),
             key=lambda p:p.stat().st_mtime)
for p in paths[-1:]:
    print(p.name,p.stat().st_size)
    for raw in p.read_text().splitlines()[-8:]:
        data=json.loads(raw)
        line=json.dumps({'time':data.get('timestamp'),'level':data.get('level'),'fields':data.get('fields')})
        line=re.sub(r'https?[^\s\"<>]+','[URL]',line)
        line=re.sub(r'(?:hf_|gh[pousr]_)[A-Za-z0-9]{15,}','[REDACTED]',line)
        line=re.sub(r'(?i)(authorization|token|credential|signature)[^,}]*',r'\1 [REDACTED]',line)
        print(line[:1200])
