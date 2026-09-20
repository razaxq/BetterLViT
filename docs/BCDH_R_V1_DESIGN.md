# BCDH-R V1 validation pilot

BCDH-R V1 is a boundary-conscious dual-head output refiner. It does not use a
boundary target, distance transform, direction map, surface loss, Hausdorff
loss, or boundary Dice loss.

## Locked architecture

- Parent: frozen CXR-BERT, LoRA disabled, FAM-EPPA V4-B decoder.
- Coarse head: complete-mask prediction from the 112 x 112 `up2` decoder
  feature, bilinearly upsampled to 224 x 224.
- Fine/base head: the existing complete-mask prediction from the 224 x 224
  `up1` feature.
- Prediction-only cues: detached coarse probability, `4p(1-p)` uncertainty,
  fine-only disagreement, and coarse-only disagreement.
- Refiner: two lightweight convolutions followed by a zero-initialized output
  projection. The bounded residual is added to the base logits with
  `delta_max=1.0`.
- Initial condition: candidate final probabilities exactly equal base
  probabilities.

## Locked objective

Both outputs use only the original complete segmentation mask:

```text
L_seg = 0.5 * Dice + 0.5 * Focal(gamma=2)
L_total = L_seg(final, target) + 0.2 * L_seg(coarse, target)
boundary_loss_weight = 0.0
```

## Paired validation protocol

- C1: identical parent with BCDH disabled.
- P6: identical parent with BCDH-R V1 enabled.
- 40 epochs, physical batch 16, seed 1219, deterministic CUDA, drop_last true.
- Test access is disabled. The chain exports only the best-checkpoint
  validation predictions at threshold 0.5.

The numeric gate requires validation macro Dice delta >= 0.002, no overall or
smallest-lesion-quartile precision loss, no smallest-lesion Dice loss, improved
boundary F1 at tolerance 2 pixels, and non-worse Brier score. Residual
distribution and train-validation gap remain mandatory manual checks.
