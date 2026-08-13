# FAM-EPPA V4-C Experiment

## Objective

V4-C tests the missing alignment component of the V4-B frequency-fusion
design. It retains the V4-B ALPF/AHPF paths and adds local-similarity-guided
semantic-flow alignment only at `up4` and `up3`. The experiment uses the same
dataset, seed, optimizer, Dice/Focal loss and validation-only threshold
protocol. Boundary supervision remains disabled.

The primary baseline is V4-B, whose full 2,113-image test result is Dice/IoU
`0.844562/0.759546` at threshold `0.5` and `0.844996/0.760273` at the
validation-selected threshold `0.52`.

## Result-driven hypothesis

V4-B learned non-trivial frequency routing. At its best checkpoint, `up4`
used ALPF/AHPF strengths `0.3096/0.0653`, while `up3` reached
`0.3940/0.2264`; all predicted filter mixtures remained normalized. This
recovered V3 performance and improved IoU slightly, but Dice was effectively
tied. The remaining limitation is that the top-down decoder is still resized
and fused at fixed coordinates.

FreqFusion combines ALPF and AHPF with an offset generator because frequency
selection alone cannot replace inconsistent or displaced features. SFNet and
FaPN independently identify fixed-coordinate fusion of adjacent feature levels
as a source of semantic misalignment. V4-C therefore isolates alignment as the
next architectural variable instead of adding another filter or loss term.

## V4-C alignment

For the shared V4-B context `Z`, eight-neighbour cosine similarities `S(Z)`,
decoder feature `D`, flow groups `g`, and a bounded flow strength `alpha_f`:

```text
S(Z)       = cosine similarity from each pixel to its 8 neighbours
raw_flow   = Conv([Z, S(Z)])
flow       = 1.5px * alpha_f * tanh(raw_flow)
D_aligned  = grouped_bilinear_warp(D, flow)
D_adaptive = D_aligned + alpha_alpf * (LP_w(D_aligned) - D_aligned)
```

The last flow-prediction layer is initialized to zero, so the initial warp is
exactly identity and V4-C initially reproduces V4-B. Four channel groups are
used, matching the best offset-group count reported by FreqFusion. Maximum
displacement is `1.5` pixels at the decoder feature resolution.

## Controlled scope

- `up4`, `up3`: V4-B plus semantic-flow alignment.
- `up2`, `up1`: unchanged V4-A fusion.
- Frequency groups: `8`; flow groups: `4`.
- Flow strength: initial `0.25`, bounded by `1.0`.
- Maximum flow displacement: `1.5` feature pixels.
- Loss: Dice/Focal (`0.5/0.5`, gamma `2.0`).
- Boundary loss: `0.0`.
- Seed: `1219`; batch size: `16`; maximum epochs: `200`.

## Diagnostics and acceptance checks

The log records per aligned stage:

- flow strength and mean/maximum displacement;
- ratio of locations displaced by more than `0.25` feature pixels;
- alignment residual standard deviation;
- skip/decoder cosine agreement before and after alignment;
- local-similarity mean and standard deviation;
- all inherited ALPF/AHPF, Haar and PLAM diagnostics.

The structural check requires exact identity alignment at initialization,
finite forward/backward values, non-zero flow-predictor gradients, normalized
filter mixtures and bounded displacement. Checkpoints must carry architecture
version `fam_eppa_v4c`.

## Evaluation protocol

The best checkpoint is selected only by validation Dice. Final evaluation
first uses all 1,429 validation images to select a threshold, then evaluates
all 2,113 test images at threshold `0.5` and at the fixed validation-selected
threshold. The primary acceptance target is to exceed V4-B
`0.844996/0.760273`; statistical interpretation must acknowledge that the
V3/V4-B Dice difference is currently negligible.

## References

- Chen et al., *FreqFusion: Frequency-aware Feature Fusion for Dense Image
  Prediction*, TPAMI 2024: <https://arxiv.org/abs/2408.12879>
- Li et al., *SFNet: Faster and Accurate Semantic Segmentation via Semantic
  Flow*: <https://arxiv.org/abs/2207.04415>
- Huang et al., *FaPN: Feature-aligned Pyramid Network for Dense Image
  Prediction*, ICCV 2021: <https://arxiv.org/abs/2108.07058>

## Runtime maintenance after Epoch 6

The first server session exposed frequent host-side gaps between CUDA kernels.
Profiling traced the dominant avoidable synchronization to per-image
GPU-to-CPU copies followed by scikit-learn IoU evaluation on every training
batch. The continuation keeps the architecture, optimizer, scheduler, loss,
batch size, seed, and deterministic cuDNN policy unchanged, while applying
runtime-only maintenance:

- vectorized per-image IoU on the active torch device;
- device-side epoch accumulation with one host snapshot every 20 batches;
- non-blocking copies from pinned DataLoader memory;
- one batched tokenization pass when each dataset is constructed.

The continuation must use a new session sourced from the complete Epoch 6
checkpoint. The original session and checkpoint remain immutable.
The server passes that source through `BETTERLVIT_RESUME_PATH`, so the tracked
configuration remains clean and an omitted variable still means from-scratch
training.
