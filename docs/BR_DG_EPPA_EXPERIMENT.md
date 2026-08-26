# Balanced-Residual DG-EPPA Experiment

## Research question

Can a balanced residual spatial gate retain the first DG-EPPA model's useful
background suppression while preserving lesion boundaries well enough to beat
both tag `839752` and the boundary-loss model without boundary supervision?

`boundary_loss_weight` remains `0.0`, so this is an architecture-only
experiment.

## Evidence from DG-EPPA v1

DG-EPPA v1 improved the default-threshold test result from
`Dice 0.837091 / IoU 0.750527` to `0.841424 / 0.755901`. However, the spatial
gates at `up4`, `up3`, and `up1` suppressed 92.81%, 90.35%, and 99.23% of
locations while almost never amplifying them. The architecture behaved mostly
as a learned low-pass filter and did not surpass the boundary-loss model.

## BR-DG-EPPA design

The second version keeps the exact low/high-frequency decomposition:

```text
skip = skip_low + skip_high
```

It changes the attention update to an explicit residual highway:

```text
output = skip
       + channel_residual * skip_low
       + spatial_residual * skip_high
```

The spatial residual has two independently bounded components:

1. A zero-mean local component redistributes high-frequency evidence across
   locations without globally erasing it. Its learned per-channel strength is
   bounded to `[0, 0.5]` and starts at `0.1`.
2. A global component permits mild sample-dependent smoothing, but its
   per-channel strength is bounded to `[0, 0.15]` and starts at `0.05`.

Low-frequency skip semantics, decoder semantics, local edges, and dilated edge
context are combined by a learned, channel-wise softmax mixture instead of
being added indiscriminately. This keeps the fusion bounded without creating a
large concatenated activation at the full-resolution stage. CXR-BERT `[CLS]`
features additionally modulate the fused spatial guide with an
identity-initialised FiLM transform. Channel descriptors are L2-normalised
before their MLP to reduce scale-driven gate saturation.

Both attention heads remain zero-initialised. Therefore BR-DG-EPPA is exactly
an identity mapping before training even though its residual strengths start
nonzero, allowing gradients to reach the attention heads immediately.

## Diagnostics and acceptance

Each validation epoch records the original gain statistics plus:

- mean of the balanced local residual, which should remain approximately zero;
- mean global residual;
- learned local and global residual strengths;
- observed minimum and maximum spatial gain;
- spatial-logit saturation ratio;
- text FiLM modulation magnitude.
- entropy and mean weights of the four guide branches.

The primary comparison uses threshold `0.5`. A secondary threshold may be
selected using only the 1,429-image validation split and must be locked before
the 2,113-image test evaluation.

The architecture must exceed DG-EPPA v1 (`Dice 0.841424`, `IoU 0.755901`) and
the stronger calibrated target remains the boundary-loss result
(`Dice 0.842278`, `IoU 0.758155`).

## Commands

```powershell
python tools/check_crossgate_experiment.py
python train_model.py
python tools/evaluate_experiment.py
```

## Design references

- Wang et al., [Residual Attention Network for Image
  Classification](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_Residual_Attention_Network_CVPR_2017_paper.html).
- Oktay et al., [Attention U-Net: Learning Where to Look for the
  Pancreas](https://arxiv.org/abs/1804.03999).
- Yang et al., [Gated Channel Transformation for Visual
  Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.html).
- Touvron et al., [Going Deeper With Image
  Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html).
