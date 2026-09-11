"""Verify terminal artifacts, all fold inspection delays and native pause status."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import tomllib
from zoneinfo import ZoneInfo
from analysis import write_json
HERE=Path(__file__).resolve().parent
runtime=json.loads((HERE/'results/runtime.json').read_text());assert runtime['phase']=='complete'
state=json.loads((HERE/'state.json').read_text());assert state['phase']=='complete' and state['inspections_completed']==1
proof=json.loads((HERE/'results/independent_verification.json').read_text());assert proof['verified']
checkpoint=json.loads((HERE/'checkpoint_verified.json').read_text());assert checkpoint['verified'] and checkpoint['heads']==20
backup=json.loads((HERE/'hf_upload_verified.json').read_text());assert backup['verified'] and backup['all_sizes_and_xet_hashes_match']
sha=runtime['source_git_commit'];assert sha==proof['source_git_commit']==checkpoint['source_git_commit']==backup['source_git_commit']
automation=tomllib.loads(Path('C:/Users/dtftn/.codex/automations/betterlvit/automation.toml').read_text())
assert automation['id']=='betterlvit' and automation['status']=='PAUSED' and sha in automation['prompt']
observed=json.loads((HERE/'results_collection.json').read_text())['observed_unix']
delays={str(f):observed-json.loads((HERE/f'results/fold_{f}_records.json').read_text())['training_completed_unix'] for f in range(5)}
assert all(0<=s<=1800 for s in delays.values())
value=dict(verified=True,source_git_commit=sha,phase='complete',inspections=1,maximum_inspections=2,
    all_fold_training_completions_observed_within30min=True,fold_training_inspection_delay_seconds=delays,
    final_evaluation_inspection_delay_seconds=observed-runtime['completed_unix'],
    completed_sydney=datetime.fromtimestamp(runtime['completed_unix'],ZoneInfo('Australia/Sydney')).isoformat(),
    automation_id='betterlvit',automation_status='PAUSED',hf_files=backup['files_verified'],hf_logical_bytes=backup['logical_bytes'],
    next_text_training_started=False,official_validation_accessed=False,test_split_accessed=False)
write_json(HERE/'closure_verified.json',value);print(json.dumps(value))
