# rs1s3407 Val replication

Source `ec3d45cc43710b239e3dadef3f99eb1e2823aa8e`; seed 3407; 80 epochs; Best 74.

| Model | IoU | Dice | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| R2 matched seed | 72.6484% | 82.3421% | 80.7497% | 87.7171% | 0.020952 |
| RS1 | 72.9237% | 82.5400% | 80.7909% | 88.1291% | 0.020824 |

IoU delta +0.2753 pp; grouped descriptive CI [-0.02129123890957089, 0.5727250751470129] pp.

This single result does not decide the two-seed replication. Complete both registered seeds regardless of the first numerical outcome. Precision is reported and is not an automatic veto. All metrics, small-area results and FP/FN counts are in paired_comparison.json.

Inspections 2/2; seconds after training end 1169.774. Test was not accessed. HF upload requires its separate verified receipt.
