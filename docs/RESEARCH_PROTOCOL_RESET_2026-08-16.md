# BetterLViT research protocol reset — 2026-08-16

## Decision

Freeze FAM-EPPA V4-B as the structural baseline. Do not add another alignment,
prototype, reliability, feature-residual, or text-conditioned frequency module.
The next falsifiable variable is whole-text modality dropout during training:

- control: `text_modality_dropout_prob = 0.0`;
- candidate: `text_modality_dropout_prob = 0.5`;
- replacement text: `No report available.`;
- validation and final inference always use the recorded text unless a named
  robustness counterfactual is being evaluated.

All other architecture, objective, optimizer, sample order, augmentation,
initialization, and stopping settings must remain paired.

## Evidence behind the reset

Historical benchmark results used validation-selected thresholds:

| Variant | Test Dice | Test IoU | Interpretation |
|---|---:|---:|---|
| V3 | 0.844993 | 0.760152 | Historical reference |
| V4-B | 0.844996 | 0.760273 | Best observed, but effectively tied with V3 |
| V4-H | 0.840597 | 0.754253 | Worse than V4-B |

V4-H completed 200 epochs and its route tensors were numerically active. A
validation-only causal audit nevertheless found that shuffling text only
inside the up4/up3 frequency route changed mean absolute prediction
probability by just `1.95e-6` and changed Dice by about `-4e-7` at the original
threshold. Disabling both routes slightly increased validation Dice by about
`3.5e-5`. V4-H therefore learned a near-global frequency bias rather than
sample-specific causal text use.

V4-B showed the opposite problem at the whole-model level. On the fixed legacy
validation set, at V4-B's original selected threshold `0.52`:

| Text condition | Dice | Change from correct text |
|---|---:|---:|
| Correct recorded text | 0.824273 | — |
| Dataset-wide deranged text | 0.639141 | -0.185132 |
| Fixed neutral text | 0.572818 | -0.251456 |

The structured prompts encode lesion laterality, count, and coarse lung region
and were checked against masks during dataset annotation. The model is thus
strongly dependent on target-adjacent prompt information. Whole-modality
dropout tests whether that dependence can be reduced without sacrificing
correct-text segmentation.

## Split protocol

The published LViT `5716/1429` train/validation folders are image-level, not
patient-level. Among filenames with recoverable BIMCV subject IDs, 434 subjects
cross the legacy boundary and 656 of 832 identifiable validation images come
from subjects already present in training.

The primary research track therefore uses the committed
`known_patient_grouped_sensitivity_v1` manifest built only from the original
7,145 train+validation pool:

- train: 5,716 images;
- validation: 1,429 images;
- identifiable pool: 4,194 images from 1,656 subjects;
- grouped validation: 839 identifiable images from 359 subjects;
- identifiable train/validation subject overlap: zero;
- anonymous `covid_N` validation: 590 images treated as singleton pseudo-groups.

Because 2,951 legacy images have no patient identifier, this is explicitly a
**known-patient-grouped sensitivity split**, not proof of complete patient
independence. The legacy folders remain unchanged for historical comparability,
but a grouped checkpoint must not treat the legacy validation folder as an
independent validation set.

## Strict paired harness

The new harness fixes sources of hidden experiment drift:

- sorted, asserted image/mask/text records;
- independent RNG streams for model initialization, training operations,
  sampling, workers, validation workers, and text dropout;
- Python, NumPy, and Torch worker seeding from `torch.initial_seed()`;
- `persistent_workers = false` so epoch-boundary resume is reproducible;
- deterministic algorithms, TF32 disabled, fixed cuBLAS workspace;
- strict checkpoint state loading;
- saved global and dedicated-generator RNG state;
- manifest, dataset-order, source commit, environment, and training-config
  fingerprint in each checkpoint;
- unique session names and refusal to reuse an output directory;
- one GPU process tree guarded by a filesystem lock.

Historical V4-B cannot be treated as strictly paired with this harness. A new
V4-B control must be trained from scratch first.

## Pre-registered sequence

1. Run grouped V4-B control (`p=0.0`) with seed `1219` from scratch.
2. Run the paired dropout candidate (`p=0.5`) with seed `1219` using the same
   source commit and manifest. If correct-text validation Dice falls by more
   than `0.005`, or neither neutral nor shuffled robustness improves, reject
   the candidate before more long runs.
3. If the first pair passes that gate, complete paired seeds `3407` and `7777`
   for both arms.
4. Select checkpoints and thresholds only from the grouped validation split.
5. Report per-seed values, paired deltas, mean and standard deviation. For
   identifiable subjects, use patient-cluster paired bootstrap; report the
   anonymous subset separately.
6. Unlock the official test benchmark only once for the final selected arm.
   Report image-level metrics for literature comparability and patient-equal /
   patient-cluster uncertainty for the identifiable subset.

## Candidate decision rules

The candidate is a robustness success only if, across the three paired seeds:

- correct-text mean Dice is no more than `0.001` below the control;
- both shuffled-text and neutral-text mean Dice improve by at least `0.02`;
- the mean correct-to-counterfactual degradation gap shrinks by at least 25%;
- no pre-registered lesion-size subgroup loses more than `0.005` Dice.

It is an accuracy success only if the correct-text mean Dice improves by at
least `0.002` and the patient-cluster paired 95% confidence interval for the
delta excludes zero. Robustness and accuracy claims must not be conflated.

## Locked constants

- architecture: FAM-EPPA V4-B;
- batch size: 16;
- maximum epochs: 200;
- loss: Dice/Focal `0.5/0.5`;
- focal gamma: `2.0`;
- boundary loss weight: `0.0`;
- checkpoint selection: validation Dice only;
- threshold selection: validation only;
- one active training process tree;
- no test access during implementation, sanity checks, or model selection.

## Relevant primary literature

- LViT: <https://openaccess.thecvf.com/content/CVPR2023/html/Li_LViT_Language_Meets_Vision_Transformer_in_Medical_Image_Segmentation_CVPR_2023_paper.html>
- Missing-modality segmentation: <https://arxiv.org/abs/1908.06683>
- SGSeg: <https://papers.miccai.org/miccai-2024/270-Paper0556.html>
- ProLearn textual reliance study: <https://openaccess.thecvf.com/content/ICCV2025/html/Ye_Alleviating_Textual_Reliance_in_Medical_Language-guided_Segmentation_via_Prototype-driven_Semantic_ICCV_2025_paper.html>
- QaTa-COV19-v2 / OSegNet: <https://arxiv.org/html/2202.10185>
