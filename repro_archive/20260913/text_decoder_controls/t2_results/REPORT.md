# T2 matched decoder control (validation only)

Source `488ef093de80df71ee77741a8c7ee7b938c7d6b5`; seed1219;80epochs;Best69.

| Model | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| R2 reused | 72.6654% | 82.4034% | 80.5086% | 88.0502% |
| T2 | 72.4663% | 82.2327% | 79.9910% | 88.4644% |

IoU delta -0.1990 pp; grouped descriptive CI [-0.5001925409230537, 0.10676770274936484] pp.

Single-seed discovery only; no Test access, stable gain or novelty claim. All small-area metrics and gradient observations are preserved.
Export-minus-checkpoint-selection IoU=1.3156875589e-10. Training uses >=0.5 while frozen export uses >0.5; reconcile any material discrepancy before performance interpretation.

Smallest-mask quartile: 358 images, area <= 2075 pixels.
| Metric | Small-mask delta |
|---|---:|
| iou | -1.0053 pp |
| dice | -0.9223 pp |
| precision | -0.9357 pp |
| recall | +0.0341 pp |

Brier: R2 0.02116154; T2 0.02148347; delta +0.00032192.
Total pixel count differences (candidate minus control): {'tp': 15685, 'fp': 54616, 'fn': -15685}. Counts do not replace per-image macro metrics.

Final inspection 2026-09-13T12:13:12.353477+10:00; 1067.924 seconds after training ended; inspections 2/2.

| Observed Train epoch | Residual RMS / feature RMS | Attention entropy | Q gradient absmax |
|---|---:|---:|---:|
| 1 | 0.000000 | 2.819544 | 0 |
| 10 | 0.165806 | 2.788838 | 0.00302568 |
| 40 | 0.197086 | 2.681667 | 0.00774273 |
| 80 | 0.188424 | 2.742480 | 0.00677604 |

These observations cover one ordinary Train batch at each of four epochs. Nonzero residuals/gradients show that the branch is active on those batches; they do not establish a beneficial causal effect or dataset-wide grounding.
