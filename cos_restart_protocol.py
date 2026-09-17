"""Four authorized additional warm-restart runs, with immutable recipes."""
import os
from paper_experiments import get_paper_experiment
from training_recipe import planned_rates, rates_equal

PROFILES = {'cr0_plam':('CR0','legacy_plam',False),
            'cr3_fsdr_race_binding':('CR3','fam_eppa_v4b',True)}


def validate_manifest(m):
    group,decoder,race=PROFILES[m['profile']]
    p=get_paper_experiment(m['profile'])
    assert m['configuration']==group and m['seed'] in (2027,3407)
    expected=dict(epochs=80,batch_size=16,image_size=224,num_workers=4,train_drop_last=True,
        loss_name='dice_focal',optimizer='Adam',weight_decay=.0001,selection_metric='iou',threshold=.5,
        augmentation_policy='legacy',lr_schedule='warm_restarts',initialization='from_scratch',
        text_use_lora=False,boundary_loss_weight=0.,decoder_fusion_mode=decoder,
        race_enabled=race,race_route_enabled=race,race_binding_repair=race,race_aux_weight=.05 if race else 0.,
        race_pe_enabled=False,auto_validation_export=True,auto_test_evaluate=True,test_split_allowed=True,
        unknown_report_policy='legacy_fallback',count_target_policy='original_p8_unchanged',
        stage1_match_fsdr_initialization=decoder=='legacy_plam')
    for k,v in expected.items():assert m[k]==v,(k,m[k],v)
    for k in ('decoder_fusion_mode','race_enabled','race_aux_weight','race_binding_repair','selection_metric','lr_schedule','loss_name','text_use_lora'):
        assert p[k]==m[k],k
    assert p.get('race_route_enabled',True)==m['race_route_enabled']
    assert rates_equal(m['planned_epoch_lrs'],planned_rates('warm_restarts'))
    assert len(m['experiment_tag'])>10 and not m.get('resume_path')
    return m

def environment(m,commit,run):
    validate_manifest(m)
    cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
    return dict(os.environ,BETTERLVIT_EXPERIMENT=m['profile'],BETTERLVIT_SEED=str(m['seed']),
        BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16',BETTERLVIT_TRAIN_DROP_LAST='1',
        BETTERLVIT_NUM_WORKERS='4',BETTERLVIT_DETERMINISTIC='1',BETTERLVIT_CUDNN_ENABLED='1',
        BETTERLVIT_VIS_FREQUENCY='100000',BETTERLVIT_GIT_COMMIT=commit,BETTERLVIT_RESUME_PATH='',
        BETTERLVIT_EPOCH_TIMING_PATH=str(run/'epoch_timing.jsonl'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(m['seed']),TOKENIZERS_PARALLELISM='false',
        TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules')
