# Decoder-Guided Multi-Scale EPPA Experiment

## Research question

Can an architectural change to LViT's skip attention outperform tag `839752`
without relying on boundary-aware supervision?

The experiment uses the original region Dice + weighted BCE objective
(`boundary_loss_weight = 0`). This isolates the contribution of the model
architecture.

## Diagnosis of PLAM and FreqEPPA

Original LViT PLAM derives a single per-pixel weight from average and maximum
channel projections of the encoder skip. It has no decoder gating signal and
therefore cannot use coarse semantic context to suppress irrelevant
high-resolution structures.

FreqEPPA improves this by splitting a skip into low- and high-frequency
components. Low frequency drives channel attention and high frequency drives
spatial attention. However, the spatial branch still sees only the skip's edge
magnitude. It cannot determine whether a high-frequency response belongs to
the target or to an irrelevant structure.

The prior DC-preserving adaptive-low-pass experiment confirmed that better
kernel normalisation alone was insufficient: its validation-selected
full-test result (`Dice 0.838848`, `IoU 0.753793`) stayed below tag `839752`.
The new experiment therefore keeps the proven frequency decomposition and
changes how spatial relevance is inferred.

## DG-EPPA architecture

Each decoder stage now calls:

```text
refined_skip = DG_EPPA(skip, upsampled_decoder, text)
```

The module contains:

1. **Frequency decomposition** — the existing Gaussian-initialised depthwise
   low-pass produces `skip_low`; `skip_high = skip - skip_low`.
2. **Text-conditioned channel gain** — pooled `skip_low` and CXR-BERT `[CLS]`
   produce a channel gain in `(0.5, 1.5)`.
3. **Decoder-guided spatial gain** — projected `skip_low`, the corresponding
   upsampled decoder feature, local 3x3 edge evidence, and dilated 3x3 edge
   context are added in a compact guide space.
4. **Frequency recomposition** —
   `skip_low * channel_gain + skip_high * spatial_gain`.

The decoder signal follows the attention-gate principle: coarse decoder
semantics decide which fine encoder features should pass through the skip.
The parallel local and dilated edge branches preserve both sharp boundaries
and slightly displaced boundary context.

The final spatial projection and channel projection are zero-initialised.
Consequently both gains equal one at step zero and DG-EPPA is exactly an
identity mapping before learning. The old unused `pix_module1..4` PLAM
instances are removed from the registered model.

`get_CTranS_config()` exposes `eppa_use_decoder_guide` and
`eppa_use_dilated_edge`. Disabling either switch keeps parameter shapes stable,
so later ablations can load compatible checkpoints without changing the module
layout.

## Diagnostics and acceptance

Every validation epoch stores per-stage:

- channel gain mean and standard deviation;
- spatial gain mean and standard deviation;
- spatial amplification ratio (`gain > 1.1`);
- spatial suppression ratio (`gain < 0.9`);
- guide feature absolute mean.

Evaluation selects a threshold only on the 1,429-image validation split, then
evaluates all 2,113 test images. The architecture must first beat tag `839752`
(`Dice 0.839025`, `IoU 0.754068` after validation calibration). The stronger
secondary target is the boundary-loss model (`Dice 0.842278`,
`IoU 0.758155`).

## Commands

```powershell
python tools/check_crossgate_experiment.py
python train_model.py
python tools/evaluate_experiment.py
```

## Primary references

- Li et al., [LViT: Language meets Vision Transformer in Medical Image
  Segmentation](https://arxiv.org/abs/2206.14718).
- Oktay et al., [Attention U-Net: Learning Where to Look for the
  Pancreas](https://arxiv.org/abs/1804.03999).
- Woo et al., [CBAM: Convolutional Block Attention
  Module](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper).
