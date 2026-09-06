# RACE-PE V1 — 2026-09-06 preregistration

Status: implementation; gain unproven. Historical P8 remains a failed screen.

## Hypothesis and scope

FAM-EPPA V4-B handles frequency-aware decoder fusion. RACE-PE separates
regional disease existence from spatial extent. In P8, a region with >=0.5%
mask occupancy trains the regional mean pixel evidence toward one; this can
encourage diffuse activation. P8 validation IoU delta was +0.002769875 with
95% image-bootstrap CI [-0.000689679, 0.006307161], precision -0.001225211.
This is a mechanistic hypothesis, not established causal attribution.

Each scale gets an independently supervised presence head on zone-pooled
features and an extent map supervised against the full mask. Mean extent is
compared only with soft mask occupancy. Presence compares with presence for
text agreement; the spatial route uses extent and region-specific agreement.
Bounded residual strength is zero initially and limited to +/-0.15.

No test mask, training label, or parser output enters the inference route.
The learned text head is used at inference. Six rectangular coordinate zones
are geometric priors, not lung segmentation or anatomical ground truth.
Keep the dataset's image-coordinate convention; augment bases with image/mask.

Text labels are positive or unknown (-1). Require explicit side and level
within a comma/and-separated clause; reject negated clauses. Ambiguous and
partial descriptions remain unknown; this intentionally trades coverage for
precision. Count supervision requires a unique explicit one/two/three cue.
This is a conservative parser, not a general clinical-language negation model.

## Objective

Main: 0.5 Dice + 0.5 Focal, gamma=2. Auxiliary weight=0.05:
0.4 pixel BCE + 0.2 regional presence BCE + 0.1 occupancy MSE
+ 0.2 (0.75 known text-zone BCE + 0.25 known count CE)
+ 0.1 positive-only text-to-visual presence consistency.
Full-mask area pooling preserves fractional small-lesion targets. Presence
is positive for nonzero pooled occupancy; invalid empty zones are masked.
No boundary loss, LoRA, TCSR, BCDH, CDRR, or Lovasz is introduced.

## Registered arms

| Profile | Mechanism |
|---|---|
| c4_race_pe_control | Frozen CXR-BERT + FAM-EPPA + Dice/Focal |
| c5_race_v1_iou | Historical RACE V1 mechanism, including historical parser |
| p9_race_pe | Full RACE-PE |
| c6_race_pe_pixel_aux | Only pixel auxiliary supervision, same 0.05*0.4 coefficient, routing off |
| c7_race_pe_aux_only | All PE auxiliary supervision, routing off |

All new arms: 80 epochs, seed1219, physical batch16, drop_last=True, same
optimizer/schedule/augmentations/split. Primary checkpoint selection is
validation per-image IoU at 0.5, excluding epochs1-5 consistently with the
historical warmup exclusion. Historical profiles keep Dice selection. The
original P8 best-Dice checkpoint is contextual evidence, not a matched new
best-IoU comparator. C5 is a fresh same-budget arm if that comparison proceeds.

Initial chain: C4 then P9, then validation exports and paired bootstrap.
C5/C6/C7 are registered for attribution, not automatically launched.
This screen alone cannot distinguish parser changes from the whole PE design;
add a matched PE-parser ablation before attributing gains to a single component.

## Gates and provenance

Single-seed screen: macro IoU delta >=0.003, macro Dice and precision >=0,
smallest-mask-area quartile Dice/recall >=0, Brier delta <=0. Report IoU CI,
all size/frequency strata, evidence inside/outside masks, presence calibration,
extent occupancy error and routing strengths before expansion.
Thresholds are prospective and never revise the failed P8 gate retroactively.

Passing this is not stable gain. After attribution, preregister at least three
paired seeds, report each delta and mean/std; distinguish seed variability
from image-bootstrap uncertainty. Lock the final method before Test access.
All pilots set TEST_SPLIT_ALLOWED=0 and AUTO_TEST_EVALUATE=0; no automatic
150-epoch extension. Every training arm requires separate 40-character commit,
tag, active manifest, clean tracked source and matching result provenance.

New run outputs use system disk, leaving the nearly-full training-data disk
unchanged. No historical checkpoints are deleted.

## References

- Regional/instance distinction: https://arxiv.org/abs/1511.05286
- IoU loss comparison, deliberately excluded from first architecture screen:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html

Presence/extent separation and auxiliary supervision have prior art. The
project-specific fusion design is a candidate contribution, not a novelty or
performance claim until related-work analysis and controlled experiments pass.
