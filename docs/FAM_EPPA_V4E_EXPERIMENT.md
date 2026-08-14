# FAM-EPPA V4-E: Deep Reliability-Calibrated PLAM Fusion

## Evidence from V4-D

V4-D completed 200 epochs and selected Epoch 159 by validation Dice. Its
validation-selected threshold was `0.554`.

| Split | Threshold | Dice | IoU |
| --- | ---: | ---: | ---: |
| Validation | `0.5` | `0.823747` | `0.725100` |
| Validation | `0.554` | `0.825316` | `0.727697` |
| Test | `0.5` | `0.838836` | `0.751675` |
| Test | `0.554` | `0.840001` | `0.754129` |

The calibrated validation Dice exceeds V4-B by `0.001042`, while calibrated
test Dice/IoU fall by `0.004994/0.006143`. This opposite movement is evidence
of validation-specific fitting rather than under-training. Direct
pixel-to-token residual injection is therefore rejected for the next
controlled experiment.

## Hypothesis

V4-B already contains a useful language-conditioned PLAM path, but its
low-frequency contribution is added at every location with one global
strength. A local PLAM error can therefore pass directly into the skip output.
V4-E tests whether local visual agreement should calibrate that established
path before fusion.

This is motivated by three recent findings:

- TGC-Net reports that raw vision-language similarity can be noisy and uses a
  lightweight calibration module to purify cross-modal correspondence.
- Vision-Language Semantic Aggregation reports that text fusion is most useful
  at the deepest visual level, where semantic density is compatible with
  language, rather than at shallow detail levels.
- MedCLIPSeg explicitly models local cross-modal uncertainty to prevent
  unreliable language evidence from being treated uniformly.

V4-E adapts only the shared principle of reliability-aware calibration. It
does not copy a foundation-model backbone, add a contrastive objective, or
change the dataset.

## Deep reliability calibration

At `up4`, projected low-frequency skip (`S`), PLAM (`P`) and decoder (`D`)
features produce four local evidence maps:

```text
c_sp = cosine(S, P)
c_sd = cosine(S, D)
c_pd = cosine(P, D)
u    = abs(c_sp - c_sd)
```

A lightweight `3x3 -> GroupNorm -> SiLU -> 1x1` calibrator predicts a bounded
spatial gate:

```text
g_p = 1 + 0.5 * tanh(Cat(c_sp, c_sd, c_pd, u))
```

The final projection is initialized to zero, so `g_p = 1` at initialization
and the complete V4-E block exactly reproduces V4-B. The same gate calibrates
both the PLAM semantic feature and its direct low-frequency residual. Its
range `[0.5, 1.5]` prevents branch deletion or unbounded amplification.

## Controlled scope

- Base architecture: FAM-EPPA V4-B.
- `up4`, `up3`: unchanged adaptive ALPF/AHPF from V4-B.
- PLAM reliability calibration: `up4` only.
- Token-localized routing: disabled at every stage.
- Semantic flow: disabled at every stage.
- Loss: Dice/Focal (`0.5/0.5`, gamma `2.0`).
- Boundary loss: `0.0`.
- Seed: `1219`; batch size: `16`; maximum epochs: `200`.

Restricting calibration to `up4` follows the deep-only semantic fusion finding
and keeps this a single-variable architecture ablation.

## Diagnostics and acceptance protocol

Training logs the gate mean, standard deviation, range, amplify/suppress
ratios, skip/PLAM and decoder/PLAM agreement, gate-agreement correlation, and
calibrated PLAM residual standard deviation. Existing Haar and ALPF/AHPF
normalization diagnostics remain unchanged.

The best checkpoint is selected only by validation Dice. All 1,429 validation
images select one threshold, which is fixed before evaluating all 2,113 test
images. The primary target is V4-B calibrated test Dice/IoU
`0.844996/0.760273`. V3 (`0.844993/0.760152`) is effectively tied and remains
a co-primary reference. V4-C and V4-D are negative ablations.

## References

- Lin et al., *TGC-Net: A Structure-Aware and Semantically-Aligned Framework
  for Text-Guided Medical Image Segmentation*, 2025:
  <https://arxiv.org/abs/2512.21135>
- Yu et al., *Vision-Language Semantic Aggregation Leveraging Foundation Model
  for Generalizable Medical Image Segmentation*, 2025:
  <https://arxiv.org/abs/2509.08570>
- Koleilat et al., *MedCLIPSeg: Probabilistic Vision-Language Adaptation for
  Data-Efficient and Generalizable Medical Image Segmentation*, 2026:
  <https://arxiv.org/abs/2602.20423>

## Final result and decision

The run stopped by early stopping after Epoch 179; the validation-best
checkpoint was Epoch 98. Validation selected threshold `0.558`.

| Model | Validation Dice / IoU | Test Dice / IoU |
|---|---:|---:|
| V4-B, selected threshold `0.520` | 0.824274 / 0.727518 | **0.844996 / 0.760273** |
| V4-E, selected threshold `0.558` | **0.825310 / 0.727844** | 0.844166 / 0.759103 |

V4-E improved validation Dice by `0.001036` but reduced test Dice by
`0.000830`. The learned gate converged to mean `0.5457`, suppressed essentially
all spatial positions, and had only `-0.0462` correlation with the agreement
signal. It therefore learned a nearly global PLAM attenuation shortcut instead
of the intended local reliability calibration. V4-E is rejected as the next
base architecture; V4-B remains the strongest generalizing reference.
