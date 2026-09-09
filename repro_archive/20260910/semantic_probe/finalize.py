"""Validate all saved patch accounting and archive a completed diagnostic locally."""
import json
import shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from analysis import SELECTORS, digest, selected_mask, summarize, write_json

HERE=Path(__file__).resolve().parent
OUT=Path('D:/BetterLViT/outputs/semantic_probe_20260910')


def main():
    state=json.loads((OUT/'state.json').read_text())
    assert state['phase']=='complete' and state['inspections_completed']==1
    folder=OUT/'results'
    runtime=json.loads((folder/'runtime.json').read_text())
    for name, meta in runtime['artifacts'].items():
        assert (folder/name).stat().st_size==meta['bytes'] and digest(folder/name)==meta['sha256']
    data=json.loads((folder/'records.json').read_text());rows=data['records']
    result=json.loads((folder/'summary.json').read_text())
    assert data['baseline_state_unchanged'] and data['baseline_per_image_max_difference']==0
    assert data['diagnostic_source_git_commit']=='5f10bb1c4d83372553f204492ffcf3bbfaab782e'
    assert not data['test_split_accessed'] and len(rows)==1429
    assert summarize(rows)==result
    with np.load(folder/'patch_scores.npz',allow_pickle=False) as archive:
        patches={key:archive[key] for key in archive.files}
    assert patches['names'].tolist()==[r['name'] for r in rows]
    assert all(patches[k].shape==(1429,196) for k in (*SELECTORS,'fp','fn'))
    for i,row in enumerate(rows):
        base=row['baseline'];gt=base['label_pixels'];tp=gt-base['fn']
        assert patches['fp'][i].sum()==base['fp'] and patches['fn'][i].sum()==base['fn']
        for sel in SELECTORS:
            mask,ix=selected_mask(patches[sel][i]);s=row['selectors'][sel]
            assert mask.sum()==40*256 and ix.tolist()==s['blocks']
            fp=int(patches['fp'][i,ix].sum());fn=int(patches['fn'][i,ix].sum())
            assert fp==s['fp_captured'] and fn==s['fn_captured']
            assert abs((fp+fn)/(base['fp']+base['fn'])-s['error_capture'])<1e-12
            assert abs((tp+fn)/(gt+base['fp']-fp)-s['oracle_iou'])<1e-12
    train=json.loads((folder/'probe_training.json').read_text());audit=json.loads((folder/'audit_records.json').read_text())['records']
    fit_names=set(train['fit_names']);audit_names={r['name'] for r in audit};val_names={r['name'] for r in rows}
    assert len(fit_names)==512 and len(audit_names)==128 and len(train['history'])==256
    assert not (fit_names&audit_names or fit_names&val_names or audit_names&val_names)
    audit_means={k:{m:float(np.mean([r[k][m] for r in audit])) for m in ('iou','dice','precision','recall')}
                 for k in ('baseline','fine_probe','coarse_probe','full_correction')}
    proof=dict(verified=True,validation_samples=1429,selector_case_checks=1429*4,
               exact_block_budget=True,fp_fn_capture_recomputed=True,oracle_iou_recomputed=True,
               independent_summary_equal=True,probe_fit_audit_validation_disjoint=True,fit_samples=512,audit_samples=128,
               probe_steps=256,audit_means=audit_means,diagnostic_source_git_commit=data['diagnostic_source_git_commit'],
               baseline_source_git_commit=data['baseline_source_git_commit'],seconds_after_completion=state['seconds_after_completion'],
               completion_check_within_30min=state['completion_check_within_30min'],proceed_to_architecture=result['proceed_to_architecture'])
    write_json(folder/'independent_verification.json',proof)
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    for p in folder.iterdir():
        if p.is_file():shutil.copyfile(p,dest/p.name)
    for name in ('state.json','inspection_1.json'):shutil.copyfile(OUT/name,HERE/'execution'/name)
    write_json(HERE/'execution'/'automation_completed.json',dict(id='r2',status='deleted',reason='Diagnostic complete; all preregistered mechanism gates failed.'))
    end=datetime.fromtimestamp(runtime['completed_unix'],ZoneInfo('Australia/Sydney')).isoformat()
    snapshot=json.loads((OUT/'inspection_1.json').read_text())
    checked=datetime.fromtimestamp(snapshot['observed_unix'],ZoneInfo('Australia/Sydney')).isoformat()
    print(json.dumps(dict(verified=True,completed_sydney=end,checked_sydney=checked,proof=proof),ensure_ascii=False))


if __name__=='__main__':main()
