# Editable architecture diagrams

Open `FSDR_RACE_architecture_20260922.drawio` in draw.io / diagrams.net.
All blocks, labels, ports and arrows are native editable objects. There are no
embedded screenshots, generated raster images, external fonts or remote assets.

## Pages

1. **LViT integration** — the two U-shaped branches; four original CNN features,
   four decoder-bound RACE routes, four FSDR decoder stages and the expanded
   `Upsample → FSDR → Concat → Conv` interface.
2. **FSDR** — Haar semantic/detail paths, original-input identity residual,
   adaptive filter mixtures on the two deep decoder stages, and the separate
   `Y` and `D′` outputs.
3. **RACE** — shared report slots and anatomical prior, scale-specific visual
   evidence, agreement gate, learned convolutional residual and identity path.
   The count head and complete-module auxiliary training are explicitly marked.

## Reading the figures

- Matching names denote the same signal. In particular, `C1–C4` at the ViT
  inputs are the original CNN features, and `Vi` is the corresponding ViT
  reconstruction. Named ports avoid long crossing lines between the two U
  branches; they are not independently learned inputs.
- Text `T` conditions ViT, RACE and FSDR. FSDR uses its CLS token internally;
  RACE uses masked mean pooling over text tokens.
- The green `up` blocks include the entire decoder stage. FSDR replaces the
  original PLAM skip fusion; it is not an additional block after PLAM.
- `C*` is `C′` with RACE enabled and `C` in an FSDR-only model.
- `D` has already been upsampled before entering FSDR. `D′` feeds both a new
  low-frequency semantic input and the decoder concatenation; it is not added
  to `Y`. Adaptive filtering is active in `up4/up3`; `up2/up1` retain the base
  semantic and detail paths with `D′=D` and `R_adaptive=0`.
- RACE changes only decoder-bound CNN skips. Its visual residual is learned
  by depthwise/pointwise convolution, not by subtracting the evidence map.
- RACE's `A` is a broadcast sample-level scalar, whereas `P` and `E` are
  spatial maps. `s` is signed, bounded by ±0.15 and initialized to zero.
- Formulas condense implementation operations. The region branch includes
  projected/normalized features, text FiLM, similarity support and its output
  convolution; channel pooling descriptors are normalized in the implementation.
  `ΣBk` in the zone-pooling formula is clamped to a minimum of one in code.

## Source and reproduction

Topology was checked against these files at source integration commit
`b71218b52240c60b2e47c25d584448f0dde2f73d`:

- `nets/LViT.py`, especially the forward path and `UpblockAttention`;
- `nets/eppa.py`, `FAMHaarEPPA` and `SpatialAdaptiveFrequencyRefiner`;
- `nets/race_fuse.py`, `RACEFuse` and `_RACERoute`.

Run `python build_diagrams.py` using Python 3. The standard-library builder
recreates the uncompressed XML and `validation.json`, verifies every edge
endpoint and checks that each vertex lies on its page. The delivered file was
also imported into the real diagrams.net editor for visual review.

To make subsequent manual edits, edit the `.drawio` file directly and save a
new revision. Running the builder again replaces manual edits in that file.
No experimental scores are presented in these architecture diagrams.
