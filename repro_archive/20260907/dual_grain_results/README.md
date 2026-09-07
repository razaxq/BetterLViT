# P11 dual-grain final result

P11 completed 80 epochs and automatic validation/Test evaluation. The primary
Test macro IoU did not improve over C4. This is one seed and an exploratory Test
comparison, not evidence of stable gain or statistically established degradation.

| Test metric | C4 | P11 | P11 minus C4 (percentage points) |
| --- | ---: | ---: | ---: |
| Macro IoU | 75.6396% | 75.5661% | -0.0735 |
| Macro Dice | 84.2989% | 84.2335% | -0.0654 |
| Macro precision | 83.9363% | 84.9476% | +1.0113 |
| Macro recall | 87.4378% | 86.3347% | -1.1031 |

The paired-case bootstrap 95% interval for Test IoU difference is
[-0.2823, +0.1346] percentage points, crossing zero. Case bootstrapping does not
measure variability across training seeds. Precision improved while recall fell.
Validation IoU increased by 0.1108 percentage points (72.2796% to 72.3905%);
that increase did not transfer to Test.

## Reproduction and evidence

- P11 source: `2fc6ab5c8e4662d741fd8b994e55b780391948ac`.
- C4 source: `add4908a0d6f702b0a10c4581725b535543829b8`.
- P11 tag: `paper-p11-dual-grain-80e-b16-seed1219-20260907`.
- Seed 1219, 80 epochs, batch 16, fixed threshold 0.5; best checkpoint selected
  by validation IoU at epoch 80. Test has 2113 cases and validation has 1429.
- Runtime provenance and complete per-case exports are adjacent JSON files.
  All stage exit statuses are zero, the chain is complete, source SHA matches,
  and tracked source is clean in the inspection snapshot.
- `collect_final_inspection.py` is the collector used for the one final SSH
  snapshot. It is archived for reproducibility; do not rerun it for this run.
- `archive_final_inspection.py` verifies the already collected local snapshot
  and archives these files. It never connects to the server.

## Prediction-based final inspection

The midpoint inspection at epoch 40 predicted completion near 21:54 Sydney.
The sole final inspection was scheduled for 22:10 on 2026-09-07.
Training completion, measured by successful `training.status` modification,
was 21:55:08 Sydney. The actual snapshot began at 22:11:18 Sydney, a delay of
970.293 seconds (16 minutes 10 seconds), within the requested 30-minute window.
Training, validation, Test, and both comparisons had completed by the snapshot.

This was inspection 2 of the allowed maximum 2. There was no persistent
connection between scheduled inspections. Automation `p11` is now PAUSED.
The frozen training source was not changed and no new training or Test run was
started during the final inspection.
