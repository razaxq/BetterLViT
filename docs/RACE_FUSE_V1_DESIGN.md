# RACE-Fuse V1 preregistration

## Research claim

RACE-Fuse is a report-anatomy consistency mechanism, not a generic attention
block. It learns compact location/count semantics from frozen CXR-BERT token
features, aligns those semantics with visual evidence across all four encoder
skip scales, and applies only bounded positive evidence routes. This is the
second architecture claim beside FAM-EPPA.

The deterministic report parser supplies weak training labels only. It is not
used to generate predictions at inference; the learned slot head is used.
Unmentioned regions are unknown for report consistency and are never treated
as lesion-negative evidence.

## Locked P8 mechanism

- text encoder: frozen `microsoft/BiomedVLP-CXR-BERT-specialized`;
- semantic slots: six left/right by upper/middle/lower locations plus a
  three-class lesion-count head;
- four visual evidence routes on raw U-Net skips (64/128/256/512 channels);
- augmentation-aware six-zone bases transformed with each image/mask;
- report prior x visual evidence x agreement gate;
- identity-preserving additive residual, zero strength at initialization,
  bounded to 0.15 per route;
- main loss: Dice/Focal 0.5/0.5;
- auxiliary weight: 0.05, composed of text-slot, visual-zone and
  positive-only report consistency terms;
- LoRA disabled and boundary loss fixed to zero.

## 80-epoch paired screen

C3 and P8 use seed 1219, physical batch 16, 80 epochs, identical data split,
augmentation, optimizer, schedule and validation exporter. C3 disables RACE;
P8 enables only the locked mechanism above. Test access is prohibited.

The candidate advances only if validation macro Dice improves by at least
0.002, overall precision does not decrease, smallest-lesion-quartile Dice and
recall do not decrease, Brier score does not worsen, and route diagnostics
show non-collapsed, spatially selective activation. A failure is documented;
it is not automatically restarted or tuned against Test.
