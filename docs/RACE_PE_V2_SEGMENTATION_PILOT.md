# V2 segmentation pilot, preregistered 2026-09-07

User authorized launch after the isolated mention-head gate passed.
Run C8 auxiliary-only, then P10 auxiliary plus routing. Both use V2 mention
targets and remove report-region consistency. Only route_enabled differs.
Each starts from scratch with frozen pretrained CXR-BERT; the diagnostic
mention checkpoint is not loaded. Seed1219, 80 epochs, physical batch16,
drop_last=True, workers4, deterministic CUDA, Val IoU checkpoint selection,
fixed threshold0.5, same original selection eligibility and optimizer schedule.
Use distinct frozen commits/tags/manifests for the arms. No Test or extension.

Reuse the completed C4 seed1219/80e validation export from commit
add4908a0d6f702b0a10c4581725b535543829b8 as the unchanged no-auxiliary baseline.
Its split and preprocessing remain the same. Export all1429 Val images after
each successful training, then produce C4-vs-C8, C4-vs-P10, C8-vs-P10 comparisons.

Primary candidate screen is P10 minus C4: macro IoU >= +0.003, no regression
in macro Dice/precision, smallest-lesion-quartile Dice/recall or Brier.
Bootstrap interval lower bound >0 is required for a positive interval claim.
The same numeric screen is recorded for all pairs without changing thresholds.
Attribution to routing additionally requires positive C8-to-P10 IoU with a
positive bootstrap interval and no precision regression. A single seed still
cannot establish stable gains; additional paired seeds require a separate plan.

Inspect mention collapse, route strengths, precision/recall and auxiliary-only
behavior after completion. The previous C4/P9 Test access remains on record;
this pilot does not read Test. Save Best/Last and logs on the system disk,
which had approximately20GB free at launch. Do not increase shared-fs storage.
