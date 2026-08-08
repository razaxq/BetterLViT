# FAM-EPPA V4-A Experiment

## Objective

This experiment tests an architectural change only. It keeps Dice/Focal loss,
sets `boundary_loss_weight = 0.0`, and trains from scratch. The comparison
baseline is PLAM-Guided Normalized EPPA (V3), whose validation-selected test
result was Dice `0.844993` and IoU `0.760152`.

## Structural hypothesis

V3 added the reconstructed PLAM feature to the raw CNN skip before EPPA. It
then separated that already mixed tensor with a trainable 3x3 low-pass kernel.
The low-pass logits converged to a uniform kernel and the multiplicative edge
branch became effectively inactive.

V4-A changes three connected design choices:

1. The raw CNN skip, PLAM reconstruction and top-down decoder feature remain
   separate inputs until EPPA.
2. A fixed orthonormal Haar transform produces exactly reconstructing low- and
   high-frequency feature maps. It has no trainable kernel to collapse under
   weight decay.
3. High-frequency information has a direct additive residual with a small
   non-zero strength floor. A learnable refinement is added to this guaranteed
   path instead of multiplying the high-frequency tensor by a scalar edge map.

The PLAM branch contributes its low-frequency reconstruction as a semantic
residual. V4-A intentionally retains the existing CLS-token FiLM and same-scale
decoder guide so later token-level, adaptive-filter and cross-scale experiments
remain separately measurable.

## Output equation

For raw skip `C`, PLAM feature `P`, decoder feature `D`, and text `T`:

```text
(C_low, C_high) = Haar(C)
(P_low, P_high) = Haar(P)
D_low           = Haar(D).low

Y = C
  + alpha_plam   * P_low
  + channel_residual(C_low, P_low, D_low, T)
  + alpha_region * region_residual(C_low, P_low, D_low, T)
  + alpha_detail * (C_high * semantic_support + detail_refinement(C_high))
```

`alpha_plam`, `alpha_region`, and `alpha_detail` are independent bounded gates.
The detail gate has a floor of `0.02`. Gate logits, biases, and normalization
parameters are excluded from L2 decay.

## Reproducibility rules

- Architecture version: `fam_eppa_v4a`
- Seed: `1219`
- Epochs: `200`
- Loss: Dice/Focal, weights `0.5/0.5`, focal gamma `2.0`
- Boundary loss: disabled
- Checkpoint resume: only from a checkpoint carrying the same architecture
  version
- Model selection: validation Dice only
- Primary test threshold: `0.5`
- Secondary threshold: selected using the validation set only

The training history records Haar reconstruction error, low/high energy ratios,
PLAM/region/detail gate strengths, residual standard deviations, semantic
support, branch energies, and PLAM/decoder agreement at all four decoder stages.

