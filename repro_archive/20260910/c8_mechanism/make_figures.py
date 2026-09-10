"""Standalone scientific plots from archived numbers; Matplotlib required."""
import argparse
import csv
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--deps',help='Optional directory containing Matplotlib dependencies')
args=parser.parse_args()
if args.deps: sys.path.insert(0,args.deps)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
out=root/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','savefig.facecolor':'white'})
rows=list(csv.DictReader((root/'results/history_paired.csv').open()))
epochs=np.array([int(r['epoch']) for r in rows])
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained',
                      gridspec_kw={'height_ratios':[2,1]})
for a,color in [('c4','#758291'),('c8','#007e87')]:
    axes[0].plot(epochs,[100*float(r[a+'_val_iou']) for r in rows],label=a.upper(),color=color,lw=1.6)
axes[0].set(ylabel='Validation macro IoU (%)',ylim=(48,74))
axes[0].legend(frameon=False,loc='lower right')
axes[0].set_title('Same schedule, mixed epoch-level advantage',loc='left',fontweight='bold')
delta=100*np.array([float(r['c8_minus_c4_val_iou']) for r in rows])
axes[1].bar(epochs,delta,color=np.where(delta>=0,'#007e87','#cf694c'),width=.85)
axes[1].axhline(0,color='#555555',lw=.8)
axes[1].set(xlabel='Epoch',ylabel='C8 - C4 (IoU pp)',xlim=(.5,80.5))
for ax in axes:
    ax.grid(axis='y',alpha=.15)
    for e in range(11,81,10):ax.axvline(e,color='#555555',alpha=.12,lw=.8)
fig.savefig(out/'training_curves.png',dpi=180)
fig.savefig(out/'training_curves.pdf')
plt.close(fig)

d=json.loads((root/'results/test_analysis.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
v=100*np.array([d['fp_iou_contribution'],d['fn_iou_contribution'],d['iou']['mean']])
axes[0].bar(['FP changes','FN / TP changes','Net IoU'],v,color=['#007e87','#cf694c','#244363'],width=.65)
for i,x in enumerate(v):axes[0].text(i,x+(.009 if x>=0 else -.014),f'{x:+.3f}',ha='center',va='bottom' if x>=0 else 'top')
axes[0].axhline(0,color='#555555',lw=.8)
axes[0].set(ylabel='IoU change (percentage points)',ylim=(-.14,.37))
axes[0].set_title('Test: error-count decomposition',loc='left',fontweight='bold')
q=d['size_quartiles'];means=100*np.array([x['iou']['mean'] for x in q]);ci=100*np.array([x['iou']['ci95'] for x in q])
axes[1].errorbar(range(1,5),means,yerr=[means-ci[:,0],ci[:,1]-means],fmt='o',color='#007e87',capsize=5)
axes[1].axhline(0,color='#555555',lw=.8)
axes[1].set(xticks=range(1,5),xticklabels=['Q1\nSmallest','Q2','Q3','Q4\nLargest'],ylabel='C8 - C4 macro IoU (pp)',xlabel='Total GT lesion area quartile')
axes[1].set_title('No confirmed size-specific gain',loc='left',fontweight='bold')
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.savefig(out/'test_mechanism.png',dpi=180)
fig.savefig(out/'test_mechanism.pdf')
plt.close(fig)
print('Saved four figure files. Source: results/history_paired.csv and test_analysis.json')
