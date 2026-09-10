"""Matched-epoch history comparison; no checkpoint/threshold selection."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    source={a:json.loads((ROOT/'evidence'/(a+'_history.json')).read_text())
            for a in ('c4','c8','p10')}
    histories={a:d['epoch_history'] for a,d in source.items()}
    assert all(len(h)==80 and [r['epoch'] for r in h]==list(range(1,81)) for h in histories.values())
    lr=[r['lr'] for r in histories['c4']]
    assert all([r['lr'] for r in h]==lr for h in histories.values())
    rows=[]
    for i in range(80):
        row=dict(epoch=i+1,lr=lr[i])
        for a,h in histories.items():
            for k in ('train_iou','val_iou','train_loss','val_loss'):
                row[a+'_'+k]=h[i][k]
            for split in ('train','val'):
                c=h[i][split+'_loss_components']
                row[a+'_'+split+'_main_loss']=c['total'] if a=='c4' else c['main']
        row['c8_minus_c4_val_iou']=row['c8_val_iou']-row['c4_val_iou']
        rows.append(row)
    summary=dict(matched_lr_all_epochs=True,
        c8_val_iou_wins=sum(r['c8_minus_c4_val_iou']>0 for r in rows),
        epochs=80,windows={},last_epoch_losses={})
    for n in (10,20,40,80):
        selected=rows[-n:]
        delta=np.array([r['c8_minus_c4_val_iou'] for r in selected])
        summary['windows']['last_'+str(n)]=dict(
            means={a:float(np.mean([r[a+'_val_iou'] for r in selected])) for a in histories},
            c8_minus_c4_mean=float(delta.mean()),wins=int((delta>0).sum()),
            warning='Correlated epochs with cyclic learning rates; not independent replications.')
    for a,h in histories.items():
        summary['last_epoch_losses'][a]={k:h[-1][k] for k in
            ('train_loss','val_loss','train_loss_components','val_loss_components')}
    summary['matched_epochs']=[r for r in rows if r['epoch'] in (1,10,20,30,40,50,60,70,80)]
    for filename,content in [('history_analysis.json',summary)]:
        (ROOT/'results'/filename).write_text(json.dumps(content,indent=2)+'\n')
    with (ROOT/'results/history_paired.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__': main()
