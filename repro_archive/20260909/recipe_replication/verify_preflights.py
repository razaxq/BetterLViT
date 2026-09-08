"""Verify independent repeats and C4/R2 parity separately within each seed."""
import json
import sys
from prepare_replication import HERE,ROOT
sys.path.insert(0,'D:/BetterLViT/recipe_replication_work')
from training_recipe import rates_equal,validate_recipe_manifest

sources=json.loads((HERE/'sources.json').read_text())
proof={}
for seed in (2027,3407):
    values={}
    for role in ('c4','r2'):
        label=role+'s'+str(seed)
        manifest=json.loads((ROOT/f'{label}_manifest.json').read_text())
        validate_recipe_manifest(manifest)
        rows=[json.loads((ROOT/f'{label}_preflight_{rep}.json').read_text()) for rep in (1,2)]
        for row in rows:
            assert row['status']=='ok' and row['seed']==seed and not row['formal_training_performed'] and not row['test_split_accessed']
            assert row['source_git_commit']==sources[label]['source_git_commit'] and row['visual_prior'] is None
            if role=='c4':assert row['training_recipe'] is None
            else:
                assert row['training_recipe']['lr_schedule']=='single_cosine'
                assert rates_equal(row['training_recipe']['planned_epoch_lrs'],manifest['planned_epoch_lrs'])
        for field in ('initial_base_sha256','input_image_sha256','output_sha256_each_step','loss_each_step'):
            assert rows[0][field]==rows[1][field]
        values[role]=rows[0]
    for field in ('initial_base_sha256','input_image_sha256','output_sha256_each_step','loss_each_step'):
        assert values['c4'][field]==values['r2'][field],(seed,field)
    proof[str(seed)]={field:values['c4'][field] for field in ('initial_base_sha256','input_image_sha256','first_output_sha256','loss_each_step')}
assert proof['2027']['initial_base_sha256']!=proof['3407']['initial_base_sha256']
result=dict(status='verified',sources={k:v['source_git_commit'] for k,v in sources.items()},
    independent_repetitions=2,temporary_steps=5,within_seed_pairing=proof,test_split_accessed=False)
(ROOT/'preflight_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result))
