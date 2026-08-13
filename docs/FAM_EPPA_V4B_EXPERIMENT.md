# FAM-EPPA V4-B Experiment

## Objective

V4-B tests whether spatially adaptive frequency selection can improve the
V4-A skip fusion without using boundary supervision. It keeps the same data,
seed, optimizer, Dice/Focal objective and threshold protocol as V3 and V4-A.
Only the low-resolution `up4` and `up3` decoder stages change.

The primary target is the V3 validation-selected test result: Dice `0.844993`
and IoU `0.760152`. V4-A reached `0.841970/0.755838` under the same calibrated
protocol.

## Structural hypothesis

V4-A uses fixed Haar decomposition at every decoder stage. It preserves detail
exactly, but its frequency response is identical at all pixels. This can blur
small lesions and preserve noisy texture indiscriminately.

FreqFusion addresses a related feature-fusion problem using a spatially varying
adaptive low-pass filter (ALPF) for low-resolution semantic features and an
adaptive high-pass filter (AHPF) for high-resolution detail. V4-B adopts this
principle, but not its expensive CARAFE/unfold implementation. Instead, each
location and channel group predicts a convex mixture of three fixed normalized
filters:

1. identity;
2. 3x3 binomial low-pass;
3. 5x5 binomial low-pass.

This is a project-specific efficient adaptation of FreqFusion, not an exact
reproduction. It preserves normalized kernels and spatially varying routing
while keeping memory suitable for the local 24 GB AMD GPU.

## V4-B fusion

For skip feature `C`, PLAM feature `P`, decoder feature `D`, and learned
per-location mixture weights `w`:

```text
context = project(C_low) + project(P_low) + project(D_low)
w_low, w_high = softmax(predict(context))

LP_w(X) = w_identity * X + w_blur3 * Blur3(X) + w_blur5 * Blur5(X)
D_adapt = D + alpha_alpf * (LP_w_low(D) - D)
C_high_adapt = alpha_ahpf * (C - LP_w_high(C))

Y = V4A(C, P, D_adapt, T) + C_high_adapt
```

Both strength parameters are bounded. `alpha_ahpf` also has a non-zero floor,
so the adaptive detail branch cannot vanish completely. The adapted decoder is
used both as EPPA's semantic guide and in the subsequent decoder concatenation;
therefore ALPF has a direct path to the segmentation output.

## Controlled scope

- `up4` and `up3`: V4-A plus adaptive ALPF/AHPF.
- `up2` and `up1`: unchanged V4-A fusion.
- Groups: `8`; context channels: `32`.
- ALPF strength: initial `0.20`, maximum `0.50`.
- AHPF strength: initial `0.08`, floor `0.02`, maximum `0.30`.
- Boundary loss: `0.0`.
- Loss: Dice/Focal (`0.5/0.5`, gamma `2.0`).

Restricting adaptive filtering to the two coarsest decoder stages reduces GPU
memory and isolates whether semantic alignment benefits before applying the
same mechanism to high-resolution stages.

## Diagnostics and acceptance checks

The training log records, separately for `up4` and `up3`:

- ALPF/AHPF strengths;
- identity, Blur3 and Blur5 mixture weights;
- kernel sums and normalized entropy;
- decoder filtering delta and skip high-frequency residual standard deviation;
- all inherited V4-A Haar, energy, branch-strength and agreement diagnostics.

Every spatial kernel must sum to one. The dedicated CPU check also verifies
finite forward/backward values and non-zero gradients for both predictors and
strength gates. The AMD integration check uses a full real-data batch.

## Reproducibility and evaluation

- Architecture version: `fam_eppa_v4b`.
- Training: from scratch, up to 200 epochs, seed `1219`.
- Checkpoint resume is allowed only when the stored architecture version is
  exactly `fam_eppa_v4b`.
- Best checkpoint is selected by validation Dice.
- Test metrics are reported at default threshold `0.5` and at a second
  threshold selected exclusively on all 1,429 validation images.
- The 2,113-image test set is evaluated only after model and threshold choices
  are fixed.

## Final outcome

Training completed 200 epochs. The best checkpoint was Epoch 198, with
validation Dice/IoU `0.824076/0.727164` at threshold `0.5`. Validation-only
calibration selected threshold `0.52`:

| Split | Threshold | Dice | IoU |
| --- | ---: | ---: | ---: |
| Validation | `0.5` | `0.824076` | `0.727164` |
| Validation | `0.52` | `0.824274` | `0.727518` |
| Test | `0.5` | `0.844562` | `0.759546` |
| Test | validation-selected `0.52` | `0.844996` | `0.760273` |

V4-B exceeds V4-A by Dice `0.003026` and IoU `0.004435`, showing that the
adaptive frequency paths are useful. Against V3, however, calibrated Dice is
effectively tied (`+0.000002`) and IoU improves only `0.000121`.

The learned filters were non-trivial and normalized. At Epoch 200, `up4`
ALPF/AHPF strengths were `0.3096/0.0653`; `up3` reached `0.3940/0.2264`.
This rules out branch collapse but also shows that frequency selection alone
has reached a plateau. The next controlled ablation is V4-C: preserve V4-B
and add the omitted local-similarity offset alignment at `up4/up3`.

## Reference

- Chen et al., *FreqFusion: Frequency-aware Feature Fusion for Dense Image
  Prediction*, TPAMI 2024: <https://arxiv.org/abs/2408.12879>
- Official implementation: <https://github.com/Linwei-Chen/FreqFusion>
