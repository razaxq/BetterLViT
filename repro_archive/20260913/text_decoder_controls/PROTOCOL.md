# T1/T2 intermediate decoder controls (registered before training)

Purpose: establish whether an additional text path improves per-image macro IoU
over equal added visual capacity. Ordinary cross-attention is a control, not an
innovation claim. No extra supervision or loss is added.

Common recipe: original R2 `9eca26de5b301099805530edbf5a1a8718bea662`, seed 1219,
80 complete epochs, batch 16, grouped Adam, Dice/Focal, single cosine from 3e-4
to 1e-6, original legacy augmentation, frozen CXR-BERT, FAM-EPPA V4-B, threshold
strictly > 0.5, checkpoint selected by validation macro IoU. T0 reuses the already
completed 80-epoch R2 seed1219 (Val IoU 72.665359%, Dice 82.403429%, Best67).
This saves a deterministic duplicate run; it is not a newly trained baseline.
Historical R2 sources and new disabled-path five-step grouped-Adam outputs must
match exactly before reuse. New T1/T2 each have their own frozen source SHA/tag.

Insertion: after up3 and before up2, feature B x 128 x 56 x 56. Queries are 3136
positions; width32, four heads, 32 context tokens. T1 context uses 4 x 8 pooled
visual tokens (explicit reshape/mean for deterministic CUDA backward); T2 uses
existing text2 B x 32 x 128 and masks padding keys. T1 never uses report length.
Each adds 16896 parameters: bias-free Q/K/V/out and two affine LayerNorms128.
Fixed residual0.1, only output weights zero initialized, no new dropout. Module
construction follows all original modules in a CPU RNG-preserving context.
Both adapters get identical seeded weights; baseline RNG and weights remain exact.

For T2, real reports retain their valid CLS/SEP keys. An empty report (no more
than CLS/SEP) produces zero residual through a finite safe softmax. The current
tokenizer uses these two special tokens; the corpus has lengths13-24. Masking
this new path does not imply padding invariance of legacy Conv1d text processing.
T2 adds gradients to the existing text projection. Unmentioned regions are not
background and no label-derived text or fixed anatomical coordinates are used.

CPU checks: counts, matched initialization, identity, masked-key invariance,
empty fallback/backward, text dependence, visual independence, early gradients.
Real Train CUDA checks: five steps, batch16, exact baseline/input/RNG parity,
actual shapes, finite forward/backward, output then Q/K/V gradients, residual
growth, original text encoder frozen, grouped optimizer, observation on/off exact
repeat. Diagnostics do not access Val or Test and are not performance results.
Formal training observes its existing first batch at epochs1,10,40,80 to record
residual RMS, attention entropy and adapter/text_module2 gradients. Observations
use detached tensors, add no forward passes or RNG draws, and cannot alter loss.

Execution order: complete T1, automatically export Best validation, verify and
archive/back up, then complete T2 with exactly the same budget regardless of T1
ranking. Never terminate a healthy run based on early ranking. Per run at most
two predictive remote status snapshots: one near15min to forecast from complete
post-warmup epochs, one predicted training end+12min, with actual end-to-check
delay recorded (target <=30min). No persistent SSH training connection.

Report T1-T0, T2-T0 and T2-T1 on all validation images at the frozen threshold;
include macro IoU (primary), macro Dice, precision/recall, small-mask IoU and
paired patient-group descriptive bootstrap95% intervals (10000 draws,seed1219).
Seed1219 is discovery only. Direction worth extending: T2 exceeds both controls
in IoU, with >=0.3 percentage point T2-T0; cross-zero intervals or small-mask
regressions must be reported, not hidden. This is not a stable-gain acceptance
gate: independent seed2027/3407 verification is needed before any stability claim.
T1/T2 negative results remain useful and do not authorize calling attention novel.
Follow with phrase binding T3 and image/text evidence control T4 only after the
matched-control diagnosis. No new Test access during this interface screen; a
frozen final candidate will need Test to support any final performance claim.

All code, configuration, observations, logs, source tags and results are to be
published to the existing GitHub repository. Completed models go additively to
the Hugging Face Bucket under their source SHA prefixes. Shared storage must
remain below 20000000000 actual bytes. Preserve Best and current RS1 Best/Last;
only separately audited cloud-verified inactive historical Last copies may be
removed. Each new launch requires more than4GB scratch free, including room for
atomic checkpoint replacement.
