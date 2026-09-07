import json,os,pathlib,subprocess,sys
p=pathlib.Path(__file__).resolve().parent
source=pathlib.Path('/root/race_pe_v2_runs/c8_p10_20260907')
env=dict(os.environ, HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',TEST_SPLIT_ALLOWED='1')
try:
 for arm in ['c8','p10']:
  original=json.loads((source/f'{arm}_validation.json').read_text())
  repo=f'/root/BetterLViT-race-pe-v2-{arm}'
  assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==original['checkpoint_git_commit']
  assert not subprocess.check_output(['git','-C',repo,'status','--porcelain','--untracked-files=no'],text=True).strip()
  (p/'status').write_text(arm+'_test_evaluating')
  env.update(RACE_EVAL_REPO=repo,BETTERLVIT_EXPERIMENT=original['experiment'])
  with (p/f'{arm}_test.log').open('w') as log:
   subprocess.run([sys.executable,str(p/'evaluate_test.py'),'--experiment',original['experiment'],'--checkpoint',original['checkpoint'],'--output',str(p/f'{arm}_test.json'),'--batch-size','16','--threshold','0.5'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((p/f'{arm}_test.json').read_text())
  assert result['checkpoint_best_epoch']==original['checkpoint_best_epoch'] and result['samples']==2113
 (p/'c4_test.json').write_bytes(pathlib.Path('/root/race_pe_test_20260907/c4_test.json').read_bytes())
 subprocess.run([sys.executable,str(p/'compare_test.py')],check=True)
 (p/'status').write_text('complete')
except BaseException:
 (p/'status').write_text('failed')
 raise
