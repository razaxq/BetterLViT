# Frozen visual prior: first diagnostic, 2026-09-08

Authorized after the 2026-09-08 research report. This is a fixed-view Train/Val
feature-readout diagnostic, not a BetterLViT result or a new supervision method.

- Source base: C4 `add4908a0d6f702b0a10c4581725b535543829b8`.
- Two pinned encoders: CXformer-S and DINOv2-with-registers-S; identical effective
  DINOv2-S configuration and eager attention. Weight revision/checksums are required.
- Input: existing C4 224 OpenCV resize, BGR-to-RGB for the external branch,
  ImageNet normalization, no histogram equalization, crop, augmentation or AMP.
- 5716 Train, 1429 Val; Test is never constructed. Full FP32 features cached on
  local scratch; the cache is for this fixed-view diagnostic only.
- Both heads: 384->64->1 pointwise GELU, 16x16 logits, nearest resize to224,
  existing equally weighted Dice/Focal. Same head seed1219, same sampler seed2220,
  AdamW lr3e-4, decay1e-4, batch16, drop_last on Train, 20 epochs, constant LR.
- Select Best only by Val macro IoU, threshold0.5. Report paired per-image metrics
  and both full histories. This head budget is not the planned BetterLViT adapter.
- Diagnose feature quality and preprocessor contract, not new segmentation SOTA.
  Choose CXformer if its Val IoU exceeds the matched natural-image encoder; if not,
  inspect the representation/contrast issue before selecting the full-run encoder.
- The proposed joint adapter is independently preflighted for exact identity,
  RNG preservation, frozen eval state, token shape and gradient flow after two steps.
- Use disconnected background execution. Maximum two inspection snapshots for
  this paired diagnostic; one launch confirmation and one predicted final check.
  No persistent SSH or completion polling. Record actual end/check time difference.

Commands (paths supplied explicitly; external weights belong outside Git):

```bash
python tools/prepare_visual_weights.py --output /root/visual_prior_models
python tools/preflight_visual_probe.py --models /root/visual_prior_models --data /root/autodl-tmp/datasets/Covid19 --output /root/visual_probe_preflight.json
python tools/probe_visual_prior.py --models /root/visual_prior_models --data /root/autodl-tmp/datasets/Covid19 --cache /root/autodl-tmp/visual_probe_cache --output /root/visual_prior_runs/probe_20260908
```

Source/weights/configs/environment must be recorded separately. No formal 80-epoch
run is automatically authorized by a probe score alone: its implementation,
deterministic whole-model preflight and matching controls must also pass. The user
has authorized proceeding through these steps without another permission prompt.
