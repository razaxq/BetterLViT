# RS1/RS2/RS3 Train-only coefficient calibration

This rule is fixed before calibration results or candidate Val results exist.

Select 32 Train filenames with the smallest SHA-256 of `regional-overlap-v1:` plus the filename. Use unchanged 224 preprocessing without augmentation, four batches of 8, seed1219 and an untrained R2 in eval mode. No optimizer update, Val or Test access is allowed.

For global/local/balanced losses on each batch compute gradients with respect to the final segmentation logit via `dL/dp * p * (1-p)`. Record norm ratios and cosine to the unchanged Dice/Focal gradient. This measures output-space gradient allocation, not parameter-gradient conflict or predicted IoU.

Let R be the maximum unweighted regional/main norm ratio across all twelve batch/mode observations. Use one common coefficient `lambda = floor(min(0.25, 0.10/R)*1e6)/1e6`. It limits each observed auxiliary logit-gradient norm to 10% of main at calibration, without tuning validation performance. Require finite R > 0 and lambda > 0. Freeze the result identically in all three profiles and formal manifests before any full training. This initial scale bound is not guaranteed throughout training.

Run the unchanged historical preflight for R2 parity; run grouped-optimizer five-step checks with lambda=0 and each candidate. Repeat candidates with scheduled Train-only telemetry inserted to prove deterministic updates and telemetry noninterference. Formal training starts from scratch and never reuses temporary preflight weights.

At epochs20/40/60/80 record the same 32 Train cases and output-gradient diagnostics, preserving parameters, buffers, parameter gradients, module modes and all RNG states. The records are scheduled inside the training job, not remote inspection polling.
