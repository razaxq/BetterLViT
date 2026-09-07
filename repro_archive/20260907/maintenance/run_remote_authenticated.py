"""Pass locally cached HF credentials via SSH stdin, never arguments or files."""
import subprocess
import sys
from huggingface_hub import get_token

token=get_token()
if not token: raise SystemExit('No cached HF credential')
script=sys.argv[1]
if not script.replace('_','').replace('.','').isalnum(): raise SystemExit('Invalid script name')
command=('source /etc/network_turbo >/dev/null 2>&1; '
         'export HF_XET_CACHE=/root/maintenance_20260907/xet_clean_cache; '
         'export HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false; '
         'export PYTHONPATH=/root/maintenance_20260907/hf_xet_143:/root/autodl-tmp/hf-bucket-client; '
         'exec /root/autodl-tmp/envs/betterlvit-paper/bin/python '
         '/root/maintenance_20260907/'+script)
proc=subprocess.run(['ssh','-i','C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519_4090d',
                     '-p','21465','-o','BatchMode=yes','-o','ServerAliveInterval=15',
                     'root@connect.westb.seetacloud.com',command],input=token+'\n',text=True)
raise SystemExit(proc.returncode)
