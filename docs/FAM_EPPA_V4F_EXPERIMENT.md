# FAM-EPPA V4-F: Mean-Preserving Semantic Prototype Aggregation

## Motivation

V4-E improved validation but regressed the independent test split because its
PLAM reliability gate collapsed into nearly uniform suppression. V4-F returns
to the V4-B frequency architecture and removes any learnable amplitude gate.
It instead asks whether a compact, image-adaptive semantic representation at
the deepest decoder stage improves generalization.

Recent language-guided medical segmentation work supports three constraints:

- semantic aggregation should reduce feature dispersion rather than inject an
  unconstrained new residual;
- language-compatible fusion is most appropriate at the deepest visual stage;
- the adaptation should be small and data-efficient for QaTa-COV19 scale.

## Mean-preserving prototype aggregation

At `up4`, let `Z` be the existing V4-B semantic tensor after skip, PLAM,
decoder and CLS-FiLM fusion. A `1x1` projection predicts four soft assignments:

```text
Q[n,k] = softmax(A(Z[n]) / temperature)
m[k]   = sum_n Q[n,k]
C[k]   = sum_n Q[n,k] Z[n] / m[k]
Z_hat[n] = sum_k Q[n,k] C[k]
Z_out  = Z + s * (Z_hat - Z)
```

The signed residual strength is bounded to `[-0.25, 0.25]` and initialized to
zero. V4-F therefore starts as an exact V4-B function. Since every assignment
row sums to one and each prototype is normalized by its assignment mass:

```text
sum_n Z_hat[n]
= sum_k C[k] sum_n Q[n,k]
= sum_k sum_n Q[n,k] Z[n]
= sum_n Z[n]
```

The reconstruction residual has zero spatial mean. Unlike V4-E, the module
cannot improve its objective by uniformly scaling the PLAM or semantic tensor.
It must learn spatial grouping structure.

## Controlled experiment

- Base: FAM-EPPA V4-B.
- Adaptive ALPF/AHPF: unchanged at `up4` and `up3`.
- Semantic prototypes: four prototypes at `up4` only.
- Temperature: `0.75`; signed maximum strength: `0.25`; initial strength: `0`.
- V4-C flow, V4-D token routing and V4-E PLAM calibration: disabled.
- Loss: Dice/Focal (`0.5/0.5`, gamma `2.0`).
- Boundary loss: `0.0`.
- Seed `1219`, batch size `16`, maximum `200` epochs.

## Diagnostics and decision rule

The run logs prototype strength, assignment and mass entropy, minimum/maximum
mass, active prototype ratio, input/reconstruction/residual standard deviation,
variance ratio, inter-prototype cosine similarity and spatial-mean preservation
error. Haar reconstruction and ALPF/AHPF kernel sums remain monitored.

The branch is acceptable only if mean error stays below `1e-5`, all four
prototypes remain active, and the independent test result exceeds V4-B. The
threshold is selected using only all 1,429 validation images and then fixed for
all 2,113 test images.

## References

- Yu et al., *Vision-Language Semantic Aggregation Leveraging Foundation Model
  for Generalizable Medical Image Segmentation*, 2025:
  <https://arxiv.org/abs/2509.08570>
- Lin et al., *TGC-Net: A Structure-Aware and Semantically-Aligned Framework
  for Text-Guided Medical Image Segmentation*, 2025:
  <https://arxiv.org/abs/2512.21135>
- Bhardwaj et al., *L3Seg: Lean Linear Layers for Language-Guided Vision
  Transformer in Medical Image Segmentation*, 2025:
  <https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/html/Bhardwaj_L3Seg_Lean_Linear_Layers_for_Language-Guided_Vision_Transformer_in_Medical_ICCVW_2025_paper.html>
