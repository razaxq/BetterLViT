"""Derive operational scripts from the completed RS1 workflow, preserving history."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'20260911/regional_supervision_execution'

def write(name,text):
    (HERE/name).write_text(text,encoding='utf-8',newline='\n')

def template(name):
    return (OLD/name).read_text(encoding='utf-8').replace("choices=('rs1','rs2','rs3')","choices=('rs1s2027','rs1s3407')")

launch=template('launch.py').replace('from verify_preflight import verify','from preflight import verify')
launch=launch.replace("if label!='rs1':", "if source['previous_label']:")
launch=launch.replace("previous_label={'rs2':'rs1','rs3':'rs2'}[label]", "previous_label=source['previous_label']")
launch=launch.replace("assert audit['source_git_commit']==backup['source_git_commit']==previous['runtime']['source_git_commit']", "assert audit['source_git_commit']==backup['source_git_commit']==previous['runtime']['source_git_commit']\n        transfer=json.loads((HERE/(previous_label+'_results/download_xet_verified.json')).read_text())\n        assert transfer['verified'] and transfer['source_git_commit']==audit['source_git_commit']\n        publication=json.loads((HERE/(previous_label+'_results/github_archive_verified.json')).read_text())\n        assert publication['verified'], 'Publish the preceding results before continuing'")
launch=launch.replace("result['experiment_tag']=source['experiment_tag']", "result['experiment_tag']=source['experiment_tag']\n    historical=json.loads((HERE.parents[1]/'20260911/regional_supervision_execution/rs1_results/runtime.json').read_text())\n    result['initial_prediction_training_seconds']=historical['training_ended_unix']-historical['started_unix']\n    result['initial_predicted_training_end_unix']=result['started_unix']+result['initial_prediction_training_seconds']\n    result['initial_predicted_training_end_sydney']=datetime.fromtimestamp(result['initial_predicted_training_end_unix'],ZoneInfo('Australia/Sydney')).isoformat()")
write('launch.py',launch)

inspect=template('inspect_run.py').replace('import math','import math\nimport time')
inspect=inspect.replace("source=json.loads((HERE/'sources.json').read_text())[args.label]", "planned=state['planned_first_check_unix'] if args.phase=='first' else state['final_check_unix']\n    assert time.time()>=planned, 'Wait for the predicted inspection time; do not poll early'\n    state['inspections_completed']+=1\n    state['inspection_attempt_started_unix']=time.time()\n    save(args.label+'_state.json',state)\n    source=json.loads((HERE/'sources.json').read_text())[args.label]")
inspect=inspect.replace("    state['inspections_completed']+=1\n    value['inspection_number']", "    value['inspection_number']")
inspect=inspect.replace("        if usable:\n", "        if not usable:\n            raise RuntimeError('No complete post-warmup epoch: preserve this check and investigate failure without repeated status polling')\n        if usable:\n")
write('inspect_run.py',inspect)

archive=template('archive_completed.py')
archive=archive.replace("ck['seed']==1219", "ck['seed']==SOURCE['seed']")
archive=archive.replace("assert len(ck['epoch_history'])==80", "assert ck['epoch']==79 and len(ck['epoch_history'])==80")
archive=archive[:archive.index("    script=Path(source['local_repository'])")]+"    print(json.dumps(dict(label=label,archived=str(destination),source_git_commit=source['source_git_commit'])))\n\nif __name__=='__main__':main()\n"
write('archive_completed.py',archive)

upload=template('upload_completed.py')
upload=upload.replace("'r2_vs_'+label+'.json'", "'paired_comparison.json'")
upload=upload.replace("stage='/root/autodl-tmp/regional_runs/hf_staging/'+prefix", "stage=str(Path(source['remote_run']).parent).replace('\\\\','/')+'/hf_staging/'+prefix")
upload=upload.replace("classification='completed_validation_only_80e_pilot'", "classification='completed_validation_only_80e_independent_seed_replication'")
# Never enter historical RS2/RS3 attribution branches for new replication labels.
write('upload_completed.py',upload)
write('verify_downloads.py',template('verify_downloads.py'))
reconcile=template('reconcile_iou.py').replace("environment(source['source_git_commit'], source['profile'])", "environment(source)")
write('reconcile_iou.py',reconcile)
print('Prepared launch, bounded inspection, static archive, metric reconciliation and HF upload scripts.')
