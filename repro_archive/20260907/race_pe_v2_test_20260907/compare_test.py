import json,pathlib,numpy as np
p=pathlib.Path(__file__).resolve().parent
for control,candidate in [('c4','c8'),('c4','p10'),('c8','p10')]:
 a=json.loads((p/f'{control}_test.json').read_text()); b=json.loads((p/f'{candidate}_test.json').read_text())
 for d in [a,b]:
  assert d['split']=='test' and d['test_split_accessed'] and d['samples']==2113 and d['threshold']==0.5 and d['selection_metric']=='iou'
 x={r['name']:r for r in a['records']}; y={r['name']:r for r in b['records']}; assert x.keys()==y.keys(); names=sorted(x)
 assert all(x[n]['label_pixels']==y[n]['label_pixels'] for n in names)
 result={'split':'test','test_split_accessed':True,'user_authorized_test':True,'samples':len(names),'threshold':0.5,'threshold_selected_on_test':False,'checkpoint_selection':'previous validation best IoU','control_git_commit':a['checkpoint_git_commit'],'candidate_git_commit':b['checkpoint_git_commit'],'overall':{}}
 for metric in ['dice','iou','precision','recall','brier']:
  av=np.array([x[n][metric] for n in names]); bv=np.array([y[n][metric] for n in names]); delta=bv-av
  rng=np.random.default_rng(1219); means=[]
  for _ in range(10000):means.append(delta[rng.integers(0,len(delta),size=len(delta))].mean())
  result['overall'][metric]={'control_mean':float(av.mean()),'candidate_mean':float(bv.mean()),'mean_delta':float(delta.mean()),'paired_image_bootstrap_95_ci':np.quantile(means,[.025,.975]).tolist()}
 (p/f'{control}_vs_{candidate}_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
