# EPPA Boundary-Loss Experiment

## Motivation

The DC-preserving adaptive-kernel experiment kept every low-pass kernel
normalised, but its calibrated full-test result (`Dice 0.838848`,
`IoU 0.753793`) remained slightly below the 839752 baseline
(`Dice 0.839025`, `IoU 0.754068`). Kernel deltas also stayed close to zero,
while the learned spatial edge projections had substantially larger weights.
The next experiment therefore keeps the winning 839752 architecture unchanged
and changes only the training objective.

## Change

The objective adds a differentiable boundary Dice term:

```text
0.9 * (0.5 * region Dice loss + 0.5 * weighted BCE)
+ 0.1 * boundary Dice loss
```

Boundary maps are computed with a differentiable 3×3 morphological gradient
(`max-pool(x) - min-pool(x)`). This directly penalises boundary displacement
while retaining 90% of the original region objective. The model architecture,
seed, batch size, optimizer, learning-rate schedule, LoRA configuration, and
dataset split remain identical to 839752.

## Safety and evaluation

- Worktree: `D:\Project\BetterLViT-eppa-boundary`
- Branch: `codex/eppa-boundary-loss`
- Shared AMD environment: `D:\Project\BetterLViT\.venv`
- Check first: `python tools\check_boundary_experiment.py`
- Train: `python train_model.py`
- Evaluate: `python tools\evaluate_experiment.py`
- Evaluate and save test masks at the default `0.5` threshold:
  `python tools\evaluate_experiment.py --save-predictions
  --prediction-threshold 0.5`

Evaluation scans thresholds only on the 1,429-image validation split, then
applies the selected threshold once to all 2,113 test images. Metric reductions
run on CPU to avoid the invalid float64 reduction observed on the Windows AMD
stack. Results must exceed calibrated test Dice `0.839025` and IoU `0.754068`
to count as an improvement.

## Result

Training completed 200 epochs. The best checkpoint was epoch 160. With the
validation-selected threshold `0.608`, the full 2,113-image test split reached
Dice `0.842278` and IoU `0.758155`, improving on the calibrated 839752
baseline by `0.003253` Dice and `0.004087` IoU.

When `--save-predictions` is enabled, binary test masks are written beside the
checkpoint session in a threshold-named directory such as
`test_predictions_threshold_0.500/`. The directory also contains a manifest
recording the checkpoint, export threshold, validation-selected threshold, and
image count.
