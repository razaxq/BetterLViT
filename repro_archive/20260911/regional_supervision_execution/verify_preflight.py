"""Require legacy parity, grouped-optimizer identity, and final-SHA CUDA repeats."""
import json
import subprocess
from remote_ops import HERE,save


def read(name):return json.loads((HERE/'preflight'/(name+'.json')).read_text())


def verify():
    sources=json.loads((HERE/'sources.json').read_text())
    legacy=read('legacy_r2');base=read('grouped_r2')
    old=json.loads((HERE.parents[1]/'20260908/recipe_execution/r2_preflight_3.json').read_text())
    assert read('checks')['status']=='ok'
    assert read('gate_checks')['status']=='ok'
    for k in ('initial_base_sha256','input_image_sha256','output_sha256_each_step','loss_each_step'):
        assert legacy[k]==old[k],('historical_r2',k)
    calibration=read('calibration')
    assert calibration['max_weighted_ratio']<=.10 and calibration['common_weight']==.128312
    assert calibration['initial_base_sha256']==base['initial_base_sha256']
    candidates={}
    for label in sources:
        repo=sources[label]['local_repository']
        unchanged=subprocess.check_output(['git','diff','--name-only',
            '9eca26de5b301099805530edbf5a1a8718bea662',sources[label]['source_git_commit'],
            '--','nets','utils.py','Load_Dataset.py','Train_one_epoch.py','training_recipe.py'],cwd=repo,text=True)
        assert not unchanged.strip(),(label,'Unexpected base model/data/objective change')
        manifest=json.loads((HERE/(label+'_manifest.json')).read_text())
        assert manifest['regional_weight']==calibration['common_weight']
        remote=json.loads((HERE/'github_sources_verified.json').read_text())
        assert remote['verified'] and remote['refs']['refs/tags/'+sources[label]['experiment_tag']]==sources[label]['source_git_commit']
        first=read(label+'_first');repeat=read(label+'_audit');disabled=read(label+'_disabled')
        for d in (first,repeat,disabled):
            assert d['status']=='ok' and d['source_git_commit']==sources[label]['source_git_commit']
            assert not d['test_split_accessed'] and not d['formal_training_performed']
            for k in ('initial_base_sha256','initial_rng','input_image_sha256','first_output_sha256'):
                assert d[k]==base[k],(label,k)
        for k in ('output_sha256_each_step','loss_each_step'):
            assert first[k]==repeat[k],(label,'repeat',k)
            assert disabled[k]==base[k],(label,'disabled',k)
        assert not first['audit_exercised'] and repeat['audit_exercised'] and disabled['regional_disabled']
        candidates[label]=dict(source_git_commit=sources[label]['source_git_commit'],base_sources_unchanged=True,repeat_identical=True,
            telemetry_noninterference=True,disabled_grouped_optimizer_exact=True,
            peak_allocated_bytes=max(first['peak_allocated_bytes'],repeat['peak_allocated_bytes']),
            steady_seconds_per_batch=first['steady_seconds_per_batch'])
    result=dict(status='ok',legacy_5_steps_exact=True,shared_initialization_exact=True,
        common_weight=calibration['common_weight'],candidates=candidates,test_split_accessed=False)
    save('preflight_verified.json',result)
    return result


if __name__=='__main__':print(json.dumps(verify(),indent=2))
