# RS1 IoU-first independent-seed execution

Working directory: `D:/BetterLViT/experiment_docs_work`. All commands below use its Python with UTF-8: `python -X utf8`. Read `PROTOCOL.md`, `sources.json`, latest `*_state.json`, and the tracker before acting. Preserve historical R2/RS1/RS2/RS3/F worktrees, caches and results.

## Scope

Run RS1 seed2027 then seed3407, 80 epochs each, from scratch, common coefficient 0.128312 and original single cosine recipe. Complete both irrespective of the first numeric result. Compare matching R2 Val exports pinned in sources.json. No new Test access in this stage. Final assessment uses two independent replication seeds; historical1219 is discovery only. `TEXT_NEXT.md` describes the subsequent controlled text work; it has not been trained.

## Frozen sources and preflight

`prepare.py` created two distinct sources/tags. `deploy.py` deploys isolated remote copies and verifies GitHub branches/tags. `preflight.py --label LABEL` runs disabled/RS1/audit repeats. `extra_preflight.py --label LABEL` separately reconciles historical legacy-Adam smoke tests with actual formal grouped-Adam R2 training. It only adapts the preflight optimizer in memory, leaving all historical sources intact. `preflight.py --verify` requires both pairs of five-step parity checks plus audit noninterference. Do not regenerate sources or re-run prepare/deploy after formal launch. `prepare_execution.py` records initial operational derivation; resulting scripts have final adjustments, so it is historical preparation, not a current update command.

## Dispatch and bounded inspections

`launch.py --label rs1s2027` dispatches and disconnects. It records the source, PID, environment, first-check time and initial historical-duration estimate. Dispatch is not a healthy-training result. Seed3407 uses the same command with its label after predecessor archival, independent verification, HF/Xet and GitHub publication. Its numerical outcome is not a launch condition.

Use the existing native heartbeat `betterlvit`, target thread `01a074f2-5c58-7221-9a58-59850e626880`. Update the same heartbeat, never add a duplicate or keep SSH open. At/after `planned_first_check_unix`, run exactly:

```
python -X utf8 repro_archive/20260912/rs1_iou_replication/inspect_run.py --label LABEL --phase first
```

This consumes check 1 of 2. Use its measured forecast to update the existing heartbeat to the Sydney `final_check_unix` (predicted training end plus about12min). Commit/push that snapshot and schedule receipt. Do not query current or old training between these two checks. At/after the final time use `--phase final` once. Report actual training-end-to-check delay and whether it is within1800sec. If complete at first check, close the run with1/2 and archive. Failed or unexpectedly still-running final snapshots are not completion: preserve the budget and failure evidence; do not invent a result or repeatedly poll. Treat the original next step as pending until a completed immutable result is available.

## Completed-run handoff

After a snapshot says `phase=complete`, these commands read static completed artifacts; they are not extra live training inspections:

```
python -X utf8 repro_archive/20260912/rs1_iou_replication/archive_completed.py --label LABEL
python -X utf8 repro_archive/20260912/rs1_iou_replication/analyze.py --label LABEL
python -X utf8 repro_archive/20260912/rs1_iou_replication/upload_completed.py --label LABEL
python -X utf8 repro_archive/20260912/rs1_iou_replication/verify_downloads.py --label LABEL
python -X utf8 repro_archive/20260912/rs1_iou_replication/publish.py --completed-label LABEL --message "Archive completed RS1 independent-seed replication"
```

If selection/export IoU differs by>=1e-7, run the existing `reconcile_iou.py --label LABEL` on that completed Best only, then re-run analysis. This retains original exports and independently explains `>=0.5` float32 training versus `>0.5` float64 export; it cannot change selected epoch, metric convention, or gate. Update the tracker with all results, actual timing and verified upload receipts before publication. HF upload uses cached credentials over stdin, short training SHA prefix, additive writes, independent path/size/Xet listing and preserved originals. Never print tokens. Do not declare an upload complete without receipts.

The uploader uses the experiment-local `upload_verified_manifest.py`: its accepted classification is specifically `completed_validation_only_80e_independent_seed_replication`, requiring successful training/Val return codes, Test=false and the matching full-SHA/short-prefix relation. This helper was isolated from the historical HF helper before the first replication upload, whose old allowlist lacked this new classification; no trained source, analysis criterion or historical helper was changed.

After2027 archival/publishing, dispatch3407 even if2027 numeric IoU is negative; update the same heartbeat to its first check. After3407 archival/publishing, run `analyze.py --aggregate`, publish the report and follow `TEXT_NEXT.md` using the selected common recipe. Do not re-open old F experiments or use the old precision veto. If there is no new live training, pause the heartbeat after the completed-stage report; any next training needs its own registered source/recipe and predictive schedule.

## Storage

`storage_cleanup.json` records deletion of only four completed inactive Last copies, each independently cloud size/Xet matched and server SHA/provenance verified. Released logical bytes: 3,385,491,304. Scratch available: 2,536,030,208 -> 5,921,529,856 immediately after cleanup; actual shared-fs bytes unchanged at18,559,782,256, below20,000,000,000. All Best and F/B caches preserved. Never re-run the completed cleanup or remove other artifacts without a new concrete file/process/cloud audit. Launch requires scratch free>4,000,000,000.

`publish.py` explicitly pushes https://github.com/razaxq/BetterLViT.git, since origin is a local bare repository. It stages only this execution folder and the tracker. Its receipt records a verified already-pushed commit; archive the receipt in the next documentation commit without pretending a self-referential receipt can certify its own commit.
