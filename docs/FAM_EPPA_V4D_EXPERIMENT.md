# FAM-EPPA V4-D: Token-Localized Semantic Routing

## Objective

V4-D tests whether the diagnostic text should guide different image regions
with different tokens. It starts from the stronger V4-B architecture, retains
adaptive ALPF/AHPF fusion at `up4` and `up3`, and removes V4-C semantic flow.
The only new architectural variable is masked pixel-to-token cross-attention.

This follows the localization motivation of TextDiff and LoG: a single pooled
text vector provides global conditioning, but it cannot explicitly associate
different diagnostic phrases with different spatial regions. V4-D adapts only
their token-level architectural idea; it adds no auxiliary localization or
boundary loss.

## Token-localized routing

For visual guide `S`, all valid CXR-BERT token embeddings `T`, pixel queries
`Q`, token keys `K`, token values `V`, and a bounded residual strength
`alpha_t`:

```text
Q = normalize(Conv1x1(S))
K = normalize(Linear(T))
V = Linear(T)
A = softmax(mask(Q K^T) * temperature)
R = Conv1x1(A V)
S_token = S + alpha_t * R
```

Padding tokens are excluded before softmax. The original CLS-token FiLM path
is preserved, allowing the new branch to add local token evidence without
discarding the established global text prior.

The output projection is initialized to zero. Consequently, V4-D begins as an
exact functional copy of V4-B and learns the token-localized residual only
when supported by the segmentation objective.

## Controlled scope

- Base architecture: FAM-EPPA V4-B.
- `up4`, `up3`: adaptive ALPF/AHPF plus token-localized routing.
- `up2`, `up1`: unchanged V4-A fusion.
- Semantic-flow alignment: disabled at every stage.
- Attention dimension: `32`; heads: `4`.
- Attention temperature: learnable, initial value `5.0`, bounded to `[1, 20]`.
- Token residual strength: initial `0.10`, bounded by `0.50`.
- Loss: Dice/Focal (`0.5/0.5`, gamma `2.0`).
- Boundary loss: `0.0`.
- Seed: `1219`; batch size: `16`; maximum epochs: `200`.

## Diagnostics and acceptance checks

Each routed stage logs:

- residual strength and attention temperature;
- normalized attention entropy and peak probability;
- CLS versus non-CLS attention mass;
- spatial variation of token attention and token residual standard deviation;
- valid-token count;
- all inherited FAM-EPPA and ALPF/AHPF diagnostics.

Structural checks require V4-B-equivalent output at initialization, padding
mask invariance, finite forward/backward values, non-zero token-branch
gradients after activation, normalized adaptive kernels, and architecture
version `fam_eppa_v4d` in checkpoints.

## Evaluation protocol

The best checkpoint is selected only by validation Dice. All 1,429 validation
images select one threshold, which is fixed before evaluating all 2,113 test
images. Results are reported at threshold `0.5` and the validation-selected
threshold. The primary target is V4-B calibrated test Dice/IoU
`0.844996/0.760273`; V3 and V4-C remain secondary baselines.

## Final outcome

Training completed all 200 epochs. Epoch 159 was selected exclusively by
validation Dice, and the validation set selected threshold `0.554`.

| Split | Threshold | Dice | IoU |
| --- | ---: | ---: | ---: |
| Validation | `0.5` | `0.823747` | `0.725100` |
| Validation | `0.554` | `0.825316` | `0.727697` |
| Test | `0.5` | `0.838836` | `0.751675` |
| Test | `0.554` | `0.840001` | `0.754129` |

Although calibrated validation Dice is `0.001042` above V4-B, calibrated test
Dice/IoU are lower by `0.004994/0.006143`. V4-D also falls below V3 by
`0.004992/0.006023` and below V4-C by `0.004014/0.005223`. The token-localized
residual therefore increases split-specific fitting without improving test
generalization. The next controlled experiment removes token routing, returns
to V4-B, and calibrates the established PLAM path from local visual agreement
at the deepest decoder stage.

## References

- Xing et al., *TextDiff: Mask-Guided Residual Diffusion Models for
  Text-Supported Medical Image Segmentation*, MICCAI 2024:
  <https://arxiv.org/abs/2407.05323>
- Liu et al., *LoG: Language-driven Object-centric Grounding for
  Text-guided Medical Image Segmentation*, 2026:
  <https://arxiv.org/abs/2607.16327>
- Li et al., *LViT: Language Meets Vision Transformer in Medical Image
  Segmentation*, IEEE TMI 2023: <https://arxiv.org/abs/2206.14718>
