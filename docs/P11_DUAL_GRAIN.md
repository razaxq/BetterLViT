# P11: dual-grain visual guides, IoU-first exploratory experiment

P11 is based on frozen C4 (`add4908a0d6f702b0a10c4581725b535543829b8`).
The four existing visual Transformer streams keep their 14x14 global tokens.
At shallow stages 1 and 2, fixed average pooling produces a separate 28x28
visual grid. Fine queries read coarse semantic keys/values, combine this context
with local depthwise convolution, and project a residual onto the existing PLAM
guide. EPPA, text encoding and the segmentation objective remain unchanged.

The hypothesis is that a coarse patch's subregions benefit from distinct semantic
responses, especially for small lesion areas with false positives. The CNN already
retains fine detail, so coarse tokenization is not an established cause of errors.
This is a practical architecture candidate, not a claim of original dual-scale
attention or a demonstrated gain. Related design precedents were reviewed in the
2026-09-07 research archive; superiority over them is not claimed.

Only the final output projection is zero initialized. No additional zero gate
blocks the branch. A CPU RNG fork preserves every common initial parameter and
the subsequent random stream. The preflight checks full common-weight parity,
initial prediction identity, within-patch variation, context dependence, and
nonzero gradients after two updates, including a deterministic real batch of 16.

## Locked protocol and interpretation

See `experiment_manifests/p11_dual_grain.json`. Training starts from scratch,
80 epochs, seed 1219, batch 16, frozen CXR-BERT, Dice/Focal 0.5/0.5, Adam
3e-4 with weight decay 1e-4, the inherited cosine schedule, and deterministic
CUDA. Best is selected by validation macro IoU; threshold stays at 0.5.
The user explicitly requested automatic Test evaluation and prioritizes Test IoU.
No Test data is used in preflight, gradients, checkpoint selection or threshold
selection. Prior repeated Test inspection makes the final comparison exploratory.

The runner completes Train -> Val export -> Test export -> paired comparisons,
with separate exit-status files and full source/runtime metadata. It never resumes
an old architecture or extends training automatically. A case bootstrap is reported
for IoU, Dice, precision and recall, but a single seed cannot prove stable gains.
Any continuation must assess IoU and reproducibility before spending more GPU time.

The historical `active_race_pe.json` and RACE launchers belong to the parent C4
history. P11 uses only its own manifest and runner:

```sh
python tools/check_dual_grain.py --output /absolute/path/preflight.json
python tools/run_dual_grain_experiment.py --output /absolute/path/new_run \
  --control-validation /absolute/path/c4_validation.json \
  --control-test /absolute/path/c4_test.json
```

Run from a clean, frozen P11 checkout with the paper Python environment and
dataset symlink. The offline HF cache and deterministic environment used by the
runner must also be supplied for preflight. Each final source commit is pushed
and tagged before training; check `runtime.json` for the full executed SHA.
