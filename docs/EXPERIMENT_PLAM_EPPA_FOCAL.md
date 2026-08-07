# PLAM-Guided Normalized EPPA + Focal Experiment

## Objective

Improve QaTa-COV19-v2 segmentation without boundary supervision. The
experiment changes both the skip-refinement architecture and the region loss,
but it does not derive or supervise any boundary target.

## Motivation

BR-DG-EPPA v2 reached test Dice/IoU `0.836398/0.750102` at threshold `0.5`
and `0.837002/0.751414` after validation-only threshold selection. Its four
guide weights remained close to `0.25`; the shallow `up2` and `up1` spatial
and text paths were nearly inactive. A static competitive mixture therefore
did not learn a useful division of labour.

LViT introduced PLAM to preserve local image features. Focal Loss reshapes
cross entropy so well-classified pixels contribute less and hard pixels receive
more optimization weight. This experiment combines those ideas without adding
an edge loss.

## Architecture

`PLAMGuidedNormalizedEPPA` replaces the static four-way softmax with two direct
residual paths:

1. A PLAM-inspired pixel-semantic path uses channel average, maximum, their
   sum, and decoder-skip cosine agreement. Text applies feature-wise affine
   modulation before the pixel head.
2. An EPPA frequency path uses local and dilated-context responses from the
   high-frequency residual. Its response is gated by the PLAM semantic support.
3. Each path has an independent zero-initialized output head. The complete
   module is exactly an identity mapping at initialization.
4. Every depthwise low-pass kernel is parameterized with a spatial softmax.
   It therefore remains non-negative and sums to one throughout training.

The reported skip/decoder/local/context values are normalized contribution
energies, not trainable static mixture coefficients.

## Objective Function

The configured objective is

`L = 0.5 * L_Dice + 0.5 * L_Focal`.

For a pixel with correct-class probability `p_t`,

`L_Focal = -(1 - p_t)^gamma * log(p_t)`, with `gamma = 2`.

Foreground and background focal terms are averaged independently, then
combined with equal class weights. This preserves the class-balancing behaviour
of the previous weighted BCE while focusing learning on hard pixels. The
boundary-loss weight is fixed at `0.0`.

## Controlled Protocol

- Dataset: QaTa-COV19-v2
- Seed: `1219`
- Batch size: `16`
- Epochs: `200`
- Optimizer and cosine-restart schedule: unchanged
- Threshold: selected only on the 1,429-image validation split
- Final report: full 2,113-image test split at `0.5` and the selected threshold

Primary references:

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [LViT: Language meets Vision Transformer in Medical Image Segmentation](https://arxiv.org/abs/2206.14718)
