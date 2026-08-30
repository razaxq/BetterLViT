# Text-Conditioned Cross-Scale Skip Router (TCSR)

## Purpose

TCSR is the second architecture contribution in BetterLViT. It is independent
from FAM-EPPA: TCSR decides **which encoder skip scales and spatial regions are
relevant to the clinical text before decoding**, while EPPA refines
low/high-frequency information inside one decoder stage.

## Placement

The router consumes the four raw CNN encoder skips `(x1, x2, x3, x4)` and the
full CXR-BERT token sequence. It runs after encoder/ViT feature extraction and
before any original PLAM, FMISeg-adapted fusion, or FAM-EPPA decoder block.
PLAM tensors remain a separate guide and are not folded into TCSR.

## Routing equations

For scale `i`, TCSR projects the skip and uses spatial mean pooling (not
`adaptive_avg_pool2d`, to preserve deterministic CUDA backward):

```text
V_i = LayerNorm(mean(P_i(E_i), H, W))
T_i = TokenAttention(query=V_i, keys/values=T, mask=text_mask)
a   = softmax_i(Score_i([V_i, T_i, V_i * T_i]))
C   = sum_i a_i * Fuse([V_i, T_i])
```

`C` is the cross-scale context shared by every skip. Each scale then predicts
a text/context-conditioned channel residual and a local spatial mask:

```text
R_i = tanh(Channel_i([C, T_i]))
M_i = sigmoid(Spatial_i(P_i(E_i), T_i, C))
E'_i = E_i + tanh(gamma_i) * a_i * M_i * (E_i * R_i)
```

The four `gamma_i` parameters are initialized to exactly zero, making `E'_i`
bit-identical to `E_i` at initialization. `tanh` bounds the learned residual.
Scale weights are initialized uniformly and always sum to one per sample.

## Registered ablations

| Paper ID | Profile | TCSR | Decoder fusion | Loss | Question |
|---|---|---:|---|---|---|
| A6 | `a6_lora_tcsr` | yes | original PLAM | Dice/BCE | Does TCSR improve the LoRA baseline without EPPA? |
| A7 | `a7_lora_tcsr_freq` | yes | FAM-EPPA V4-B | Dice/BCE | Are TCSR and EPPA complementary under the A2 objective? |
| A8 | `a8_lora_tcsr_freq_focal` | yes | FAM-EPPA V4-B | Dice/Focal | Does the combined architecture retain the A4 objective gain? |

Every formal run must receive its own full Git commit and immutable tag. The
training protocol remains 150 epochs, batch size 16, seed 1219, deterministic
execution, `drop_last=True`, boundary loss zero, and automatic validation-only
threshold selection followed by Test evaluation.

## Diagnostics and acceptance checks

Training checkpoints persist whether TCSR is enabled and its routing settings.
Each validation epoch records scale weights, their sum/entropy, spatial-mask
means, effective residual gates, and token-attention entropy. The standalone
`tools/check_tcsr.py` check verifies:

- exact identity and unchanged shapes at zero-gate initialization;
- finite masked-token routing, including a defensive fully padded row;
- scale weights summing to one;
- deterministic repeated forward execution;
- non-zero gradients for the gates and all routing branches after opening the
  residual path.
