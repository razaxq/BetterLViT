"""Adapt the existing verified additive backup to this completed Train-only run."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE.parent/'text_head_screening/upload_completed_local.py').read_text()
source=source.replace("posthoc=json.loads((HERE/'posthoc_collapse.json').read_text());assert posthoc['checkpoint_load_and_hash_verified']",
    "checkpoint=json.loads((HERE/'checkpoint_verified.json').read_text());assert checkpoint['verified'] and checkpoint['heads']==18 and checkpoint['steps_per_head']==512")
source=source.replace('49905dbbdc644454a37a4a49098db0d5df5fe75a','c7080ea82ecaca0e1f33df880d168e3eb7cbc6dd')
source=source.replace('pilot-r2-text-residual-b-v1-20260911','diagnostic-text-optimizer-d1-20260911')
source=source.replace("('manifest.json','PROTOCOL.md','split.json','REPORT.md','posthoc_collapse.json','diagnose_collapse.py')",
    "('manifest.json','PROTOCOL.md','REPORT.md','NEXT_PLAN.md','derived_analysis.json','checkpoint_verified.json','verify_checkpoint.py')")
source=source.replace('completed_train_internal_frozen_head_pilot','completed_train_only_optimizer_diagnostic')
source=source.replace('steps_per_head=1024','steps_per_head=512,heads=18').replace('hf_text_head_','hf_text_optimizer_')
(HERE/'upload_completed_local.py').write_text(source,encoding='utf-8',newline='\n')
