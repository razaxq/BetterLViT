# CDRR V1 validation pilot

CDRR V1 is a Cross-Scale Detail Reliability Refiner. It is independent of
FAM-EPPA: EPPA fuses decoder features, while CDRR restricts output-logit
correction to spatially sparse, cross-scale reliable detail sites.

## Evidence motivating the design

The P6 BCDH diagnostic used 1,429 validation images and did not train or access
Test. Relative to the C1 control, P6's uncorrected base head lost 0.018400
macro Dice and 0.055832 precision. The final BCDH residual recovered most of
that loss, but 90.52% of pixels still had a negative correction on average and
the highest local-detail quartile received the weakest recovery. This indicates
auxiliary-gradient interference plus a global confidence-shrink shortcut.

## Locked architecture

- Parent: frozen CXR-BERT, LoRA disabled, FAM-EPPA V4-B decoder.
- Coarse head: complete-mask prediction from the 112 x 112 `up2` feature.
- Gradient isolation: the coarse feature and all refiner inputs are detached;
  the coarse auxiliary loss trains only the coarse head and cannot distort the
  segmentation trunk.
- Reliability score: combines base uncertainty, coarse/fine disagreement,
  cross-scale local-detail agreement, and detail shared across adjacent scales.
- Spatial support: only the highest-scoring 15% of pixels per image can be
  corrected; the residual is exactly zero elsewhere.
- Balanced correction: support-weighted residual logits are centered before a
  bounded `tanh`, preventing a global one-direction logit shift.
- The residual projection is zero initialized, so the initial final output is
  exactly identical to the C2 control output.

## Locked objective

```text
L_seg = 0.5 * Dice + 0.5 * Focal(gamma=2)
L_total = L_seg(final, target) + 0.1 * L_seg(coarse, target)
boundary_loss_weight = 0.0
```

Both terms use only the complete segmentation mask. No boundary, distance,
direction, surface, Hausdorff, or derived edge target is created.

## Paired validation protocol

- C2: identical parent with CDRR disabled.
- P7: identical parent with CDRR V1 enabled.
- 40 epochs, physical batch 16, seed 1219, deterministic CUDA, drop_last true.
- Test access and automatic Test evaluation are disabled.

The screen requires macro Dice delta >= 0.002, no overall or smallest-lesion
precision loss, no smallest-lesion Dice loss, improved tolerance-2 boundary F1,
non-worse Brier, and no Dice/precision loss in the highest quartile under both
Laplacian energy and normalized local-detail scores. Mechanism review must also
confirm 15% support, exact zero outside support, and two-direction residuals.
