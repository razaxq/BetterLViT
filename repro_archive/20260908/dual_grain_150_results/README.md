# P11 completed 150 epochs: epoch 80 remains best

The continuation completed epochs 81 through 150 successfully. None of these
epochs exceeded epoch 80's validation macro IoU, so the predeclared selection
rule retained the original epoch-80 Best. The automatic Test evaluation therefore
used that exact original checkpoint, with the original evaluator source.
All per-case Test results match the previous 80-epoch export exactly.

| Test metric | P11 after 80 epochs | Selected result after 150 epochs | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Macro IoU | 75.5661% | 75.5661% | 0.0000 |
| Macro Dice | 84.2335% | 84.2335% | 0.0000 |
| Macro precision | 84.9476% | 84.9476% | 0.0000 |
| Macro recall | 86.3347% | 86.3347% | 0.0000 |

Training longer did not improve the selected result. The all-zero paired
bootstrap interval reflects identical predictions from the same checkpoint;
it is not evidence of stability across seeds. This continuation was requested
after viewing the 80-epoch Test result. C4 at 80 epochs is only a historical
reference, not a matched-budget architectural ablation. No further training
or Test evaluation was launched during this final inspection.

## Training and checkpoint provenance

- Continuation source: `c724a62001f6c3b809cb12eee78f3bee79bdb60a`.
- Tag: `paper-p11-150e-resume-b16-seed1219-20260907`.
- Selected checkpoint and evaluator source: `2fc6ab5c8e4662d741fd8b994e55b780391948ac`.
- Selected checkpoint SHA-256: `405d0cdced5f85adfcfbc76190cac5c201ebe3a5224dd087060e32a94fbae51f`.
- Seed 1219, batch 16, fixed threshold 0.5; 2113 Test and 1429 validation cases.
- `selection.json` records 150 completed epochs and selection across all 150
  validation epochs. The exporter correctly retains `epochs: 80` and the parent
  SHA because those fields describe the selected checkpoint's original metadata.
  This does not indicate that the continuation stopped at epoch 80.
- The final training log includes epoch 150. All five stage exit statuses are
  zero; chain status is complete and tracked continuation source is clean.
- The epoch-150 validation IoU shown in the rounded training log is 0.7223,
  below the selected epoch-80 export of 0.7239046276. No Test result for the
  epoch-150 Last checkpoint is claimed.

## Final scheduled inspection

Training completed at **2026-09-08 02:43:03.665 Australia/Sydney**, using the
successful training status file's modification time as completion evidence.
The single final snapshot began at **03:00:39.425**, a delay of **1055.760 seconds
(17 minutes 36 seconds)**, within the requested 30-minute window.

This was inspection 2 of the allowed maximum 2. Automation `p11` was set to
PAUSED after the snapshot. No persistent SSH connection or additional polling
was used. Adjacent JSONs preserve runtime, selection, per-case exports,
comparisons, status timestamps and the final local monitoring state.

`archive_snapshot.py` checks only the already collected local snapshot. The
collector used for this inspection is archived under
`../../20260907/dual_grain_150_launch/collect_final_inspection.py`.
