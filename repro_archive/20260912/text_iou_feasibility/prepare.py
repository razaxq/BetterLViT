"""Freeze a Train-only, no-update feasibility audit before reading its outcomes."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent.parent / '20260911'
B = OLD / 'text_head_screening'
D = OLD / 'text_optimizer_diagnosis'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')

rows = json.loads((B / 'split.json').read_text())['records']
old = [r['index'] for r in json.loads((B / 'results/train_diagnostics.json').read_text())[0]['records']]
fit = [r['index'] for r in rows if r['eligible'] and r['partition'] == 'fit']
assert len(fit) == 1930 and len(old) == 32 and set(old) <= set(fit)
new = sorted(set(fit) - set(old), key=lambda i: hashlib.sha256(('text-e1-v1:' + rows[i]['name']).encode()).hexdigest())[:128]
selection = dict(all_eligible_fit=fit, original32=old, additional128=new,
                 selection_rule='sha256(text-e1-v1: + filename), exclude original32; first128',
                 independent_validation=False)
write('selection.json', selection)
for name in ('heads.py', 'analysis.py'):
    (HERE / name).write_bytes((B / name).read_bytes())
cache = json.loads((B / 'results/cache_manifest.json').read_text())
lookup = set(cache['unique_texts'])
semantic = [i for i in old + new if rows[i]['source_text'] != rows[i]['canonical'] and rows[i]['source_text'] in lookup]
manifest = dict(phase='E0_E1_readonly_Train_feasibility', updates=0, threshold=.5, threshold_operator='>',
    b_directory='/root/text_head_b_49905dbb', d_directory='/root/text_optimizer_d_c7080ea8',
    baseline_source_git_commit='9eca26de5b301099805530edbf5a1a8718bea662',
    b_source_git_commit='49905dbbdc644454a37a4a49098db0d5df5fe75a',
    d_source_git_commit='c7080ea82ecaca0e1f33df880d168e3eb7cbc6dd',
    b_hashes={n:sha(B/n) for n in ('split.json','results/cache_manifest.json','results/train_diagnostics.json')},
    d_hashes={n:sha(D/n) for n in ('results/heads.pt','results/train_diagnostics.json')},
    selection_sha256=sha(HERE/'selection.json'), heads_sha256=sha(HERE/'heads.py'),
    fit_count=sum(r['partition']=='fit' for r in rows), eligible_fit_count=len(fit),
    e1_count=len(old)+len(new), semantic_changed_cached_indices=semantic,
    candidate_top_fraction=.05, batch_size=16,
    old_internal_holdout_accessed=False, official_validation_accessed=False, test_split_accessed=False,
    oracle_is_deployable=False, no_automatic_training_or_architecture_pass=True)
write('manifest.json',manifest)
print(json.dumps(dict(e0_n=len(fit), e1_n=len(old)+len(new), semantic_control_n=len(semantic))))
