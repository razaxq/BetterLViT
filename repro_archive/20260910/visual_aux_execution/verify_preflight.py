"""Launch gate: baseline parity, shared randomness and telemetry noninterference."""
import json
from remote_ops import HERE,save


def read(name):return json.loads((HERE/'preflight'/(name+'.json')).read_text())


def verify():
    sources=json.loads((HERE/'sources.json').read_text())
    base=read('dev_control');legacy=read('dev_legacy_r2')
    old=json.loads((HERE.parents[1]/'20260908/recipe_execution/r2_preflight_3.json').read_text())
    assert read('dev_checks')['status']=='ok'
    assert read('gate_tests')['status']=='ok'
    for k in ('initial_base_sha256','input_image_sha256','output_sha256_each_step','loss_each_step'):
        assert old[k]==legacy[k],k
    pairs={}
    for label in ('s1','s2'):
        a,b=read(label+'_first'),read(label+'_audit')
        for d in (a,b):
            assert d['status']=='ok' and d['source_git_commit']==sources[label]['source_git_commit']
            assert not d['formal_training_performed'] and not d['test_split_accessed']
            for k in ('initial_common_sha256','initial_rng','input_image_sha256','first_output_sha256'):
                assert d[k]==base[k],(label,k)
        for k in ('output_sha256_each_step','loss_each_step','main_loss_each_step','initial_aux_sha256'):
            assert a[k]==b[k],(label,k)
        assert not a['audit_exercised'] and b['audit_exercised']
        pairs[label]=dict(sha=sources[label]['source_git_commit'],repeat_and_audit_identical=True,
            shared_initialization_and_rng_match=True,peak_allocated_bytes=max(a['peak_allocated_bytes'],b['peak_allocated_bytes']))
    assert read('s1_first')['initial_aux_sha256']==read('s2_first')['initial_aux_sha256']
    disabled=read('s2_no_aux_loss')
    assert disabled['status']=='ok' and disabled['aux_loss_disabled']
    assert disabled['source_git_commit']==sources['s2']['source_git_commit']
    for k in ('output_sha256_each_step','loss_each_step'):
        assert disabled[k]==base[k],k
    result=dict(status='ok',original_r2_training_sha=old['source_git_commit'],
        legacy_5_steps_exact=True,formal_optimizer_aux_disabled_5_steps_exact=True,
        identical_s1_s2_aux_initialization=True,candidates=pairs)
    save('preflight_verified.json',result)
    return result


if __name__=='__main__':print(json.dumps(verify(),indent=2))
