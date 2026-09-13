# M1 / M2 visual-value regrouping discovery

This continues the user's authorized search for a second structural innovation
complementary to EPPA. It follows the closed T1/T2 controls and frozen T2
interventions. The previous phrase/null-match entry condition did not pass.

Read PROTOCOL.md for the new, independently registered hypothesis and all gates.
The implementation has matching visual/text anchor variants and image-only
values. Neither variant is a confirmed innovation. No new Test access is allowed.

Frozen source SHAs, tags and paths: sources.json. Source deployment and GitHub
resolution: *_deployment.json and github_sources_verified.json. Preflight proof:
preflight_verified.json when present. Actual launch: *_launch.json and *_state.json
when present; absence means not dispatched. Never infer completion from a healthy
preflight or an early epoch. Completed metrics require *_results/validation.json,
independent_verification.json and completed runtime provenance.

Execution order: M1 then M2, 80 epochs each. After M1 completes, archive_completed,
analyze, upload_completed, verify_downloads, publish --completed-label m1;
then launch m2 regardless of M1's point estimate. Each script takes --label.
First health inspection about15 minutes after dispatch, then one predicted final
inspection. Use inspect_run.py and its local state guards; do not add manual
training status queries. Four ordinary Train batches record branch activity.

After M2 is fully verified and archived, assess_discovery.py applies the frozen
gates. A discovery pass only permits further seeds and mechanism ablations.
The final research objective remains unfulfilled until supporting evidence and
prior-art differentiation are obtained.

Storage cleanup is separate from evidence certification: backup_historical.py
preserves historical completed checkpoints without claiming fresh evaluations.
storage_cleanup.py requires fresh matching cloud hashes, complete Last metadata,
a retained sibling Best, and no active GPU/training/file references.

## Dispatch snapshot

M1 dispatched at2026-09-13 19:25:59 Australia/Sydney, PID364010. This is a
process dispatch receipt, not a completed health check or a training result.
First inspection registered at19:41 in native heartbeat betterlvit-m1-m2,
attached to current thread01a099db-3ddf-7df1-acbc-51b8f06ea9ae. M2 is queued
by the continuation workflow and has not been dispatched. Initial M1 training
end estimate is2026-09-14 00:13:41; replace it with the first-inspection forecast.

Four historical pilot backups (24files) were verified before removing only
their inactive local Last copies. All corresponding Best files remain. Scratch
free6647136256 bytes at launch; shared18559782256 bytes. Do not repeat cleanup.

## First inspection verified

At2026-09-13 19:42:22 Australia/Sydney, M1 had completed4/80 epochs, with
logged training progress epoch5 batch160/357. Source and manifest matched;
GPU100%,17404MiB; no fatal error in the saved log tail. This is a health
snapshot, not a formal result. Epochs2-4 averaged215.057984seconds. Predicted
training end2026-09-14 00:13:13; the only final inspection is scheduled at00:26
and verified against native app settings. Inspection budget used1/2. No extra
training status access is permitted before that appointment. M2 remains queued.
