"""Derive an immutable context-control runner from original R2 execution code."""
from pathlib import Path
root=Path('D:/BetterLViT/text_decoder_work')
text=(root/'tools/run_recipe_experiment.py').read_text(encoding='utf-8')
text=text.replace('import sys\n','import sys\nimport shutil\nimport torch\n')
text=text.replace('from training_recipe import validate_recipe_manifest','from training_recipe import validate_recipe_manifest, planned_rates, rates_equal')
text=text.replace("'active_recipe.json'","'active_decoder.json'")
text=text.replace('    validate_recipe_manifest(manifest)', '''    validate_recipe_manifest(manifest)
    assert manifest['profile'] in ('t1_decoder_visual','t2_decoder_text')
    assert manifest['seed']==1219 and manifest['adapter_parameters']==16896
    assert manifest['decoder_context_mode']=={'t1_decoder_visual':'visual','t2_decoder_text':'text'}[manifest['profile']]
    assert not list((repo/'Covid19').glob('**/*.pth.tar')), 'Never retrain a used worktree'
    assert shutil.disk_usage(repo).free>4_000_000_000
    assert int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])<20_000_000_000''')
text=text.replace("'recipe_training.lock'","'decoder_training.lock'")
text=text.replace("        BETTERLVIT_EPOCH_TIMING_PATH=str(args.run/'epoch_timing.jsonl'),", "        BETTERLVIT_EPOCH_TIMING_PATH=str(args.run/'epoch_timing.jsonl'),\n        BETTERLVIT_DECODER_OBSERVATIONS_PATH=str(args.run/'decoder_observations.jsonl'),")
text=text.replace("        state.update(phase='validation', best_checkpoint=str(bests[0]))", """        metadata=[]
        for path in (bests[0],bests[0].with_name('last_model-BetterLViT.pth.tar')):
            ck=torch.load(path,map_location='cpu',weights_only=True)
            assert ck['source_git_commit']==commit and ck['seed']==manifest['seed'] and ck['epochs']==80
            assert ck['decoder_context']['mode']==manifest['decoder_context_mode']
            if path.name.startswith('last_'):
                assert ck['epoch']==79 and len(ck['epoch_history'])==80
                assert rates_equal([h['lr'] for h in ck['epoch_history']],planned_rates('single_cosine'))
            metadata.append(dict(path=str(path),bytes=path.stat().st_size,source_git_commit=commit,
                seed=ck['seed'],epoch=ck['epoch'],epochs=ck['epochs'],best_epoch=ck['best_epoch'],
                history_rows=len(ck['epoch_history']),decoder_context=ck['decoder_context']))
            del ck
        write(args.run/'checkpoint_metadata.json',metadata)
        state.update(phase='validation', best_checkpoint=str(bests[0]))""")
(root/'tools/run_decoder_experiment.py').write_text(text,encoding='utf-8',newline='\n')
