# FSDR + complete RACE: final paper snapshot

The final model is displayed as **ours**. FSDR replaces PLAM; complete RACE
includes routing and binding-repaired auxiliary supervision as one module.

## Primary single-cycle cosine comparison

QaTa-COV19-v2: Train 5716 / Val 1429 / Test 2113; seeds 1219, 2027, 3407.
All four groups use 224px, batch 16, 80 epochs, frozen CXR-BERT with 32 tokens,
no LoRA, Adam with weight decay 1e-4, cosine 3e-4 to 1e-6, original augmentation,
and Dice/Focal weights 0.5/0.5 (gamma 2). RACE auxiliary weight is 0.05,
with report/visual/agreement weights 0.4/0.4/0.2. Val macro IoU selects Best;
the fixed Best is evaluated on Test using probability > 0.5, per-image macro.

| Configuration | Test IoU (%) | Test Dice (%) |
|---|---:|---:|
| LViT-PLAM, neither module | 75.4775 ± 0.0851 | 84.0251 ± 0.0760 |
| FSDR replacing PLAM, no RACE | 75.9833 ± 0.1511 | 84.5031 ± 0.0755 |
| PLAM + complete RACE | 75.6962 ± 0.1068 | 84.1928 ± 0.0688 |
| FSDR + complete RACE (ours) | 76.2262 ± 0.0456 | 84.7112 ± 0.0348 |

Values are mean ± sample SD across all three seeds. The combined model gains
0.7487 IoU percentage points over the matched PLAM control. This is not a
comparison with the unchanged official LViT pipeline, nor a significance claim.

- [All seeds, paired deltas and limits](results/stage1_overall_20260915/FINAL_REPORT.md)
- [Machine-readable sources and checkpoint hashes](results/stage1_overall_20260915/FINAL_RESULTS.json)
- [Warm-restart comparison, reported separately](results/cos_restart_overall_20260918/FINAL_REPORT.md)
- [Baseline and split audit](results/plam_baseline_audit_20260916/AUDIT.md)

Recoverable patient identifiers overlap between Train and Val (434 IDs).
No recoverable Test identifier overlap was found, but anonymous cases prevent
a claim of fully patient-independent splits. Test had prior research access.

## Text-guided external comparisons

| Method | Test IoU (%) | Test Dice (%) | Origin |
|---|---:|---:|---|
| MMI-UNet | 75.4783 | 83.9707 | Official QaTa checkpoint |
| GuideDecoder / LanGuideMedSeg | 75.2936 | 83.8836 | Project-split retraining, seed 1219 |
| DD-CMD | 76.2994 | 84.7533 | Project-split retraining, seed 1219 |

These use different training recipes and are not matched module ablations.
MMI-UNet's official training filename list was not independently verified.
The fixed eight qualitative cases are user-selected, not a population estimate.

- [External protocols and results](results/text_guided_baselines_20260920/FINAL_REPORT.md)
- [MMI full Test evidence](results/external_methods_20260920/full_test/REPORT.md)

## Provenance

The implementation integration starts at `561b8fa9489b71d9c3c222b7f4e70971c8556116`.
Each archived result retains its original runtime SHA, tag and checkpoint hash.
This integration does not retrain or re-evaluate models. Machine paths and local
gallery URLs in historical records identify the original workspace and are not
public downloads. [Import receipts](results/IMPORT_PROVENANCE.json) record exact
hashes of the copied result snapshots. Weights and datasets are not bundled here.

FSDR is an alias of the existing implementation; `eppa` state-dict keys and
historical architecture IDs remain unchanged. See [FSDR](FSDR.md).
