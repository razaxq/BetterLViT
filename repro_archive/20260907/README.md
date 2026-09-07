# Reproducibility archive — 2026-09-07

This archive preserves completed diagnostic/evaluation exports and the exact
standalone analysis scripts used during RACE-PE development. Model weights and
training artifacts belong in the Hugging Face Bucket, not Git.

## Source mapping

| Experiment | Source commit | Source branch | Status |
|---|---|---|---|
| C4 | add4908a0d6f702b0a10c4581725b535543829b8 | paper/c4-race-pe-control | 80 epochs, Val and authorized Test complete |
| P9 | 8129c1f039ed77f78e70305aca0bb9708b3b56b1 | paper/p9-race-pe | 80 epochs, Val and authorized Test complete; failed screen |
| C8 | 21606e02c2d64ae950c0f243f55163f58cbedf83 | paper/race-pe-v2-c8 | 80 epochs, Val and authorized Test complete |
| P10 | 2f33a71219ed2beb339db45d702ecc36a931e0c8 | paper/race-pe-v2-p10 | 80 epochs, Val and authorized Test complete; routing unsupported |

These are completed exploratory runs, not new 150-epoch formal ablations.
At threshold 0.5, Test macro IoU is C4 0.7563962019, P9 0.7553509599,
C8 0.7586117021, P10 0.7580904188. C8's paired case-bootstrap IoU interval
versus C4 crosses zero; no stable or multi-seed gain is established.

## Reproduction

Check out the full source commit from the table; use its experiment manifest,
dataset split and launch scripts. The `paper/race-pe-v2-development` branch
contains the V2 learnability probe and paired launcher. Test evaluator scripts
are preserved under `race_pe_test_20260907` and `race_pe_v2_test_20260907`.
They import model code from `RACE_EVAL_REPO`, select profiles through
`BETTERLVIT_EXPERIMENT`, and accept explicit checkpoint/output arguments.
Training used Python 3.12.3, torch 2.9.1+cu128, a 4090D, deterministic settings,
seed 1219, batch 16, frozen CXR-BERT, 224x224 input and threshold 0.5.

Historical wrapper scripts retain their original absolute server paths to
document what actually ran. On another server, adapt those path variables and
dataset/cache environment settings; do not assume paths point to the same data.
No credentials are included. Research documents also retain local evidence
paths where an original conversation export is not publishable.

The current research decision is in
`research_20260907/研究建议_历史复核版.md`; earlier recommendation priorities
are superseded. The historical V4-B–H table uses an older protocol and must
not be directly ranked against the C4/C8/P10 experiments above.

`server_operations` contains archived operational helpers. Read the paths and
preconditions before use; cleanup helpers are historical procedures, not an
instruction to repeat deletion. The 2026-09-07 storage operation is separately
audited before any new deletion.
