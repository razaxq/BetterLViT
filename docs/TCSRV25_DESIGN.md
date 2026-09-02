# TCSR V2.5: supervised-local routing without focal loss

## Motivation

P1-P3 validation evidence showed that coarse-to-fine routing increased false
positives on the smallest lesion quartile. P2/P3 also applied the residual over
large spatial regions. V2.5 keeps P4's sparse, single-hop `x3 -> x2` inference
path but makes its training support explicitly local.

## Non-focal objective

Both the no-TCSR control and V2.5 candidate use the same objective:

`0.5 * Dice + 0.5 * Tversky(FP=0.7, FN=0.3)`

The higher false-positive coefficient directly targets the measured
over-segmentation. No focal term is constructed for either profile.

## Supervised-local TCSR term

The ground-truth mask is resized to the routed `x2` scale. A three-pixel
morphological gradient creates a boundary target. The auxiliary loss is:

`0.02 * warmup * (mask_boundary_dice_loss + 0.5 * residual_leakage)`

where `residual_leakage` is the fraction of absolute TCSR residual energy
outside the target boundary. The weight warms up linearly over five epochs.

Labels are never inputs to the model. They are used only after the forward pass
to compute the training-only auxiliary loss. In evaluation mode V2.5 is
bit-identical to V2.4 and needs no mask.

## Validation design

1. `c0_frozen_freq_tversky`: no TCSR, establishes the paired non-focal control.
2. `p5_tcsrv25_local_tversky`: same protocol and objective, with V2.5 enabled.

Both are 40-epoch, batch-16, seed-1219 deterministic validation-only pilots.
Test evaluation is prohibited until the staged acceptance criteria pass.
