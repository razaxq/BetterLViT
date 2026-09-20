# P8 on R2: frozen matched discovery protocol

Prepared 2026-09-14. Goal: assess whether the original RACE-Fuse route can support a second architecture contribution beyond FAM-EPPA. Neither the R2 recipe nor a report parser repair counts as that contribution.

## Fixed protocol

QaTa-COV19-v2 existing Train/Val split; 5716 Train and 1429 Val images. Frozen CXR-BERT, no LoRA, FAM-EPPA V4-B, legacy synchronized augmentation, Dice/Focal plus original P8 auxiliary loss weight 0.05. Adam, weight decay 0.0001, batch 16, 80 epochs, seed 1219, deterministic CUDA. Single cosine includes 0.0003 at epoch 1 and 0.000001 at epoch 80. Select Best by Val per-image macro IoU, threshold strictly greater than 0.5. This selection protocol differs from historical P8's Dice selection.

1. Existing R2 control: source `9eca26de5b301099805530edbf5a1a8718bea662`, profile `r2_single_cosine`. Its original provenance must be checked before a paired claim.
2. P8R2 (`p8_r2_original`): original P8 V1 routing and supervision, with the fixed R2 recipe above.
3. P8B2 (`p8_r2_binding`): identical to P8R2 except explicitly covered side/level report clauses replace the first six supervision slots. Unknown grammar retains original P8 fallback. Count targets, zero/one slot BCE, evidence pooling, region bases, loss weights and initialization are unchanged. This is a binding-repair control, not a new architecture. Zero is a report non-mention target under the inherited BCE, not verified disease absence.

Both profiles are implemented in shared source; each formal run has its own immutable manifest, full Git SHA and experiment tag. The trained model does not parse raw reports at inference. The repair may alter training targets but must not add GT-derived inference information.

## Reporting and continuation

Report all contrasts, not just the best of these runs: P8R2 minus R2, P8B2 minus P8R2, and P8B2 minus R2. Primary metric is macro IoU. Also report macro Dice/Precision/Recall, Brier, FP/FN totals and predeclared GT-area quartiles (rank by area then filename, ceil boundaries). Quartiles are descriptive and cannot become post-hoc selection gates.

These two runs establish recipe transfer and repair attribution; neither can pass a second-innovation claim alone. Continue a route candidate to matched from-scratch visual/auxiliary controls only if its Val macro IoU improvement versus R2 is at least 0.30 percentage points and paired 95% interval lower bound exceeds zero. Report 10000 paired image-bootstrap resamples with RNG 20260914. Use patient-cluster intervals instead if a verified patient-ID mapping is available; never infer patient identities from filename prefixes. Image bootstrap is descriptive and cannot establish seed stability. If there is no verified patient mapping, record that limitation explicitly.

Route-specific evidence must then exceed an auxiliary/parameter-matched visual control under a separately frozen protocol before spending on seeds 2027 and 3407. Recheck three-seed mean and seed-by-seed differences; no stable gain claim from this discovery seed alone. Do not lower thresholds after seeing results. An unsuccessful transfer is reported and redirects design rather than being hidden by the repair contrast.

No Test during these discovery runs. Successful training chains an automatic Best Val export. Test requires the later frozen candidate/control protocol, with disclosure that this dataset's Test has prior project access. Do not treat four fixed-model intervention modes as candidate models or select their best score.

## Preflight and run provenance

Before launch: clean worktree, tag resolves to HEAD, manifest validation, semantic/count/fallback checks, identical native original-P8 frozen diagnostic, real-Train CUDA forward/backward determinism and finite gradients, available GPU and scratch disk. Record complete environment and source metadata. Store runs under `/root/autodl-tmp`, not shared storage. `tools/run_p8_restart.py` writes runtime state and automatically exports Val after training. Preparation is not a completed run.
