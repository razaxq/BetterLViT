# TCSR V2: Text-Conditioned Spatial Cross-Scale Skip Router

## Why V2 is necessary

TCSR V1 did not establish a positive architecture contribution under the
locked QaTa-COV19-v2 protocol.  Its strict frozen-text PLAM comparison was:

| Experiment | Router | Test macro Dice / IoU @ validation threshold |
|---|---|---|
| B0 | none | 0.840582 / 0.756348 |
| A6 | TCSR V1 | 0.836082 / 0.749756 |

V1 therefore changed Dice by -0.004500 and IoU by -0.006591.  The result does
not prove that all text-conditioned skip routing is harmful; it identifies
three weaknesses in the tested mechanism:

1. a global softmax forced all four scales to compete for one unit of routing
   budget even though fine boundary features and coarse semantic features can
   both be useful;
2. cross-scale information was reduced to a pooled vector, so no spatial
   structure actually moved between scales;
3. the residual was only a multiplicative reweighting of the original skip,
   and the exact-zero gate prevented the routing branch from receiving a full
   gradient on the first optimization step.

The A4/A9 paired LoRA ablation also shows that LoRA is not the mechanism to
recover this gap: disabling LoRA changed selected-threshold macro Dice from
0.845191 to 0.843717.  TCSR V2 therefore keeps CXR-BERT frozen and changes only
the skip router.

## V2 mechanism

For each projected skip `P_i(E_i)`, V2 aligns only the immediately adjacent
scales to the target resolution.  A finer neighbour is reduced by fixed 2x
average pooling; a coarser neighbour is expanded by nearest-neighbour
upsampling.  Both operations are compatible with the deterministic CUDA
training requirement.

```text
N_i = Fuse_i([P_i(E_i), Align(P_{i-1}(E_{i-1})),
                         Align(P_{i+1}(E_{i+1}))])
q_i = LayerNorm(mean(N_i, H, W))
T_i = MaskedTokenAttention(q_i, K(T), V(T))
```

The attended text applies a bounded FiLM transformation to the spatial
cross-scale consensus.  Each scale owns an independent sigmoid confidence;
there is no cross-scale softmax competition.

```text
N'_i = N_i * (1 + 0.5 tanh(gamma_i(T_i)))
             + 0.5 tanh(beta_i(T_i))
c_i  = sigmoid(Confidence_i([q_i, T_i, q_i * T_i]))
m_i  = sigmoid(Spatial_i(N'_i))
R_i  = Message_i(N'_i) + E_i * tanh(Channel_i([q_i, T_i]))
E'_i = E_i + s_i * c_i * m_i * R_i
```

`s_i` is independently learned and bounded to `(0, 0.5)`.  It starts at 0.05
instead of exactly zero, keeping the initial perturbation small while allowing
the visual, text, spatial, channel and neighbour-fusion branches to receive
gradients from the first backward pass.

## Separation from EPPA

TCSR V2 exchanges and selects encoder skip information *between resolutions*
before decoding.  FAM-EPPA V4-B remains responsible for frequency-aware
refinement *inside each decoder stage*.  V2 deliberately contains no wavelet,
edge or frequency filtering, so the two architecture claims remain separable.

## Registered A8 test

`a8_tcsrv2_freq_focal` is a one-factor comparison against completed A9:

| Factor | A9 control | A8 candidate |
|---|---|---|
| Text encoder | frozen CXR-BERT | frozen CXR-BERT |
| LoRA | false | false |
| Decoder fusion | FAM-EPPA V4-B | FAM-EPPA V4-B |
| Objective | Dice/Focal | Dice/Focal |
| TCSR | disabled | V2 |
| Epochs / batch / seed | 150 / 16 / 1219 | 150 / 16 / 1219 |

The Test set is evaluated only after training.  Threshold 0.5 is primary;
the secondary threshold is selected using validation data only.  Macro
Dice/IoU remain the paper-facing metrics.  Micro metrics are internal
diagnostics only.

## Acceptance checks before formal training

- shapes unchanged and all outputs finite;
- initial per-scale residual RMS ratio below 8%;
- repeat forwards bit-identical under deterministic execution;
- non-zero finite gradients for residual strength, visual projection,
  neighbour fusion, token attention, confidence, FiLM, channel, spatial and
  message branches on the first backward pass;
- output changes when valid text changes;
- changing a fine skip changes the adjacent routed scale;
- two full-model deterministic batch-16 CUDA forward/backward preflights;
- `text_use_lora=false` and zero LoRA parameter tensors.

## Research basis

- LViT motivates clinical-text guidance for medical segmentation:
  https://arxiv.org/abs/2206.14718
- Attention U-Net motivates suppressing irrelevant skip regions with gates:
  https://arxiv.org/abs/1804.03999
- Dual Cross-Attention motivates explicit multi-scale skip dependencies:
  https://arxiv.org/abs/2303.17696
- BiFPN motivates learnable multi-scale feature fusion without a single fixed
  contribution from every scale: https://arxiv.org/abs/1911.09070
- FiLM motivates feature-wise conditioning from text:
  https://arxiv.org/abs/1709.07871

