# Frozen visual prior: matched 80-epoch segmentation pilot

Implementation follows the approved 2026-09-08 research report. This is a plain
additive adapter baseline, not a claim of a novel fusion algorithm.

The selected encoder must first have a useful fixed-view Train/Val probe. Keep
C4's architecture, loss, augmentations, training schedule and frozen CXR-BERT.
Inject 41,280 trainable projection parameters at x3 after down2 and before
downVit2. The external model stays frozen/eval, processes the same augmented
224 image, uses patch tokens only, and has no auxiliary supervision.

Profiles are p12_visual_prior (CXformer weights), p12_visual_natural (matched
DINOv2 weights, only if selected by the diagnosis), and c9_visual_random (same
encoder configuration with fixed random weights). Do not sweep all profiles on
Test. The first full pilot should be the chosen pretrained candidate. A failed
C4 gate does not justify spending another 80 epochs on its random control.

Before launching, create a separate source commit and experiment tag for each
run. Each frozen worktree needs experiment_manifests/active_visual.json with:

```json
{
  "profile": "p12_visual_prior",
  "epochs": 80,
  "seed": 1219,
  "batch_size": 16,
  "selection_metric": "iou",
  "threshold": 0.5,
  "test_split_allowed": false,
  "auto_test_evaluate": false,
  "lora": false,
  "boundary_loss": 0.0,
  "initialization": "segmentation_from_scratch_frozen_external_weights"
}
```

Use the proper profile and describe initialization accurately in C9's manifest.
The runner starts from scratch and explicitly clears any inherited resume path.
It records source SHA and external weight hashes, trains in the background, then
exports the entire validation split from the selected Best checkpoint. The Test
evaluator additionally requires TEST_SPLIT_ALLOWED=1 and matching provenance;
this remains disabled during the pilot.

Run tools/preflight_visual_joint.py against each frozen source/profile, twice.
Require identical outputs/losses across the repetitions, initial C4 output
246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7,
frozen encoder gradients absent, and trainable adapter gradients after two steps.
The development checks at 5d7f4ee passed for both CXformer and random controls.
These checks used temporary optimization on one Train batch; those weights must
never initialize a formal run.

Start tools/run_visual_experiment.py with --run and --models pointing outside
the repository. The worktree needs its datasets symlink to the existing locked
dataset. Do not alter batch size or precision only for the candidate.

tools/compare_visual_validation.py validates pairing/provenance and computes the
registered C4 gate: IoU gain >=0.003, paired-image bootstrap95% lower bound>0,
Dice point estimate>=0 and lowest-GT-area-quartile IoU point estimate>=0. It also
supports the separate C9 random-feature check. A single seed is a screening result;
cross-seed stability requires paired seeds1219/2027/3407 and final Test reporting.

Use no more than two brief inspections per training run. The first captures
startup and enough real timing to predict completion; reserve the second for
after completion, targeting no more than30 minutes after training ends. Do not
hold SSH connections open or repeatedly poll. Once a run is complete, the same
authorized task may analyze results and start the next passing stage, with a new
prediction and its own two-inspection budget. An inaccurate prediction must be
reported; it does not authorize an extra hidden inspection.
