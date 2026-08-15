# FAM-EPPA V4-G: Mean-Preserving Local Reliability Residual

## Failure-driven hypothesis

V4-E and V4-F exposed two different shortcut solutions:

- V4-E used its unconstrained mean gate to suppress PLAM almost globally.
- V4-F kept four active slots but assigned every pixel uniformly, so all four
  centers became identical and the branch reduced to one-center smoothing.

V4-G returns to the best-generalizing V4-B and removes both mechanisms. It
tests whether local skip/PLAM/decoder agreement is useful only when any global
amplitude change is mathematically removed.

## Zero-mean reliability correction

At `up4`, normalized skip (`S`), PLAM (`P`) and decoder (`D`) features produce
the same local evidence maps used in V4-E:

```text
c_sp = cosine(S, P)
c_sd = cosine(S, D)
c_pd = cosine(P, D)
u    = abs(c_sp - c_sd)
```

A small `3x3 -> GroupNorm -> SiLU -> 1x1` network predicts `r`. The spatial
signal and both feature corrections are explicitly centered:

```text
q = tanh(r) - mean_hw(tanh(r))
R_sem = P_sem * q - mean_hw(P_sem * q)
R_out = P_low * q - mean_hw(P_low * q)
Z_out = Z_v4b + s * R_sem
Y_out = Y_v4b + s * R_out
```

The signed strength `s` is bounded to `[-0.25, 0.25]` and initialized at zero,
so V4-G begins as an exact V4-B function. Both corrections have zero spatial
mean per image and channel. A constant reliability prediction becomes exactly
zero, making V4-E's global attenuation shortcut impossible by construction.

## Controlled scope

- Base architecture: FAM-EPPA V4-B.
- Adaptive ALPF/AHPF: unchanged at `up4` and `up3`.
- Mean-preserving reliability residual: `up4` only.
- V4-C flow, V4-D token routing, V4-E gate and V4-F prototypes: disabled.
- Loss: Dice/Focal (`0.5/0.5`, gamma `2.0`).
- Boundary loss: `0.0`.
- Seed `1219`, batch size `16`, maximum `200` epochs.

## Diagnostics and acceptance

Training logs reliability strength, raw and centered signal statistics,
positive/negative balance, semantic/direct PLAM correction standard deviation,
per-channel mean error, and correlations with PLAM and decoder agreement.
Existing Haar and ALPF/AHPF normalization diagnostics remain unchanged.

Acceptance requires both mean errors below `1e-5`, non-zero spatial signal
variation, and complete test Dice/IoU above V4-B `0.844996/0.760273`. The
threshold is selected using only all 1,429 validation images and then fixed for
all 2,113 test images.
