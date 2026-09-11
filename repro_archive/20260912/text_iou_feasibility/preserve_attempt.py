"""Preserve the failed numerical-identity audit, then allow a separate source run."""
from pathlib import Path
import json
import base64
from control import remote
from analysis import write_json
HERE=Path(__file__).resolve().parent
receipt=json.loads((HERE/'launch.json').read_text())
assert receipt['source_git_commit']=='5e98ac651a497fa1b3ee7de3949d3f3b0d0d8ce0'
assert json.loads((HERE/'collection.json').read_text())['runtime']['phase']=='failed'
target=HERE/'attempt_v1';target.mkdir(exist_ok=False)
# Fetch completed partial artifacts by exact known file names, without polling.
result=remote('ROOT='+repr(receipt['remote_directory'])+'\n'+'''
from pathlib import Path
import base64,hashlib,json
root=Path(ROOT);out={}
for name in ('e0_per_image.json','e0_summary.json','runtime.json'):
    raw=(root/'results'/name).read_bytes()
    out[name]=dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
print(json.dumps(out))
''')
for name,value in result.items():(target/name).write_bytes(base64.b64decode(value['data']))
for name in ('launch.json','launch_attempt.json','collection.json','reproduction_debug.json'):
    (HERE/name).replace(target/name)
write_json(target/'cause.json',dict(failed_source_git_commit=receipt['source_git_commit'],train_updates=0,
    cause='requires_grad_(False) changed floating-point inference path; same parameter flag as D reproduces all metrics exactly',
    controlled_first16=True,binary_metrics_unchanged=True,strict_reproduction_requirement_preserved=True,
    retry_changes='Restore D parameter flags with no_grad inference; same cache, weights, samples, thresholds and no updates'))
print(json.dumps(dict(preserved=str(target),partial_artifacts=len(result))))
