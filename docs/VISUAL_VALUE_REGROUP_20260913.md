# Visual-value regrouping: bounded second-innovation discovery

## Decision and hypothesis

T1 and T2 did not beat R2 on validation. Frozen T2 interventions showed that its
text branch is useful within T2 and already sensitive to phrase/location binding.
Those results did not support the previous phrase/null-match proposal's entry
condition. This is a separate exploratory hypothesis, not a revival of that gate.

Hypothesis: report-conditioned exchange of existing visual evidence can improve
segmentation without injecting text vectors as decoder feature values. Text is
still used normally by all existing LViT/EPPA paths. Only the new branch changes.

## Frozen mechanism

Insert after up3, before up2: X has B x 128 x 56 x 56 elements. Four heads,
width eight each, 32 anchors, 16,896 trainable parameters, residual scale 0.1.
Let Q = Wq LN(X), K = Wk LN(context), V = Wv LN(X), S = Q K^T / sqrt(8).
A = softmax over anchors(S), B = softmax over image locations(S).
Masked anchors have zero weight in both directions. P = B^T V and
X_out = X + 0.1 Wo (A P - V). Wo starts at zero; initialization preserves
the base network weights and RNG state. No extra loss or trainable gate.
Spatially constant visual values have a zero centered message in exact arithmetic.
This does not guarantee absence of false positives or clinical grounding.

M1 (`m1_visual_regroup`): context is a deterministic 4 x 8 pooled visual grid.
M2 (`m2_text_regroup`): context is existing text2, B x 32 x 128. Padding is
masked; CLS/SEP-only or empty reports disable this branch. All values come
from image features. Both models use identical parameters and initialization.

EPPA remains responsible for its existing local frequency/detail fusion;
this branch tests global visual grouping conditioned by the report.

## Discovery experiment and gates (registered before training)

Run M1 then M2, each from scratch, seed1219, 80 epochs, batch16. M2 is not
cancelled based on M1's score. Reuse R2 (`9eca26de5b301099805530edbf5a1a8718bea662`)
as historical structural baseline. Keep frozen CXR-BERT, no LoRA, FAM-EPPA V4-B,
Dice/Focal, boundary loss0, legacy augmentation, grouped Adam and R2 single
cosine schedule unchanged. Each experiment receives a distinct commit and tag.

Select Best by the existing validation IoU rule (eligible from epoch6).
After successful training automatically export Best on all1429 Val images,
batch16, strict probability >0.5, per-image macro metrics. Reconcile any
>= versus > selection discrepancy before interpretation. No new Test access.

Advance only if ALL conditions hold:
1. M2 minus R2 macro IoU >=0.003 (0.30 percentage points), and the paired
   grouped descriptive 95% bootstrap IoU interval has a lower bound >0.
2. M2 minus M1 macro IoU >0, with grouped descriptive CI lower bound >0.
3. M2 macro Dice and smallest-mask-quartile macro IoU do not regress against
   either R2 or M1. The quartile is fixed by Val ground-truth area, not predictions.
4. Provenance, actual LR history, frozen encoder, numerical health and Best
   checkpoint selection are verified. Report precision, recall, Brier and FP/FN
   alongside IoU; they are not substitutes for the primary metric.

These are discovery gates, not statistical proof across training seeds. A pass
licenses matched additional seeds and mechanism ablations (uniform anchors,
report mismatch, removing the centered subtraction); it does not confirm novelty
or permit Test-driven redesign. A failure closes this version without retuning
gates. If both models improve but M2 does not beat M1, evidence favors visual
aggregation rather than a text-specific contribution.

## Prior art and novelty boundary

The visual pixel-to-region-to-pixel principle exists in Object-Contextual
Representations (OCR): https://arxiv.org/abs/1909.11065 . Language-conditioned
mutual reconstruction in medical segmentation is explored by RecLMIS:
https://arxiv.org/abs/2404.02845 . Therefore attention, prototype aggregation and
visual-only values cannot be claimed as independently novel primitives.
This experiment tests whether report anchors plus a centered visual message
offer a useful contribution complementary to EPPA. A performance pass still
requires a sharper comparison with prior art and isolated mechanism evidence.

## Execution and stopping

CPU behavioral checks and five actual Train-batch CUDA optimizer steps precede
launch. Verify R2/disabled exact five-step parity; candidate initial parity;
matched candidate initial weights; nonzero Q/K/V gradients after the first step;
masking/empty fallback; visual-value-only data flow; observer noninterference.
No Val/Test data are used in these preflights.

One remote health check about15 minutes after each launch. Predict completion
from ordinary epoch2-4 timings and schedule one final inspection12 minutes after
the forecast, aiming to inspect within30 minutes of actual training completion.
Maximum two training inspections per run; no continuous SSH polling. If a final
inspection is premature or a run fails, record the exception explicitly rather
than silently exceeding this bound or launching a duplicate.

Archive and verify completed M1 evidence, Best/Last and source on Hugging Face
and publish the GitHub evidence before M2. Preserve model/cache contents; any
space reclamation requires completed historical checkpoints, a sibling Best,
fresh cloud hash verification and no active file references. Shared storage
must stay below20,000,000,000 bytes; launch requires >4,000,000,000 scratch bytes.
