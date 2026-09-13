# M2 matched decoder control (validation only)

Source `8930f4b39b155d927c1556f396fb4919ca5c7c94`; seed1219;80epochs;Best67.

| Model | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| R2 reused | 72.6654% | 82.4034% | 80.5086% | 88.0502% |
| M2 | 72.5972% | 82.3084% | 79.6986% | 88.9426% |

IoU delta -0.0682 pp; grouped descriptive CI [-0.35207369319736254, 0.21978352559868258] pp.

Single-seed discovery only; no Test access, stable gain or novelty claim. All small-area metrics and gradient observations are preserved.
Export-minus-checkpoint-selection IoU=-8.25490564793e-10. Training uses >=0.5 while frozen export uses >0.5; reconcile any material discrepancy before performance interpretation.

Smallest-mask quartile: 358 images, area <= 2075 pixels.
| Metric | Small-mask delta |
|---|---:|
| iou | -0.7793 pp |
| dice | -0.7677 pp |
| precision | -1.7827 pp |
| recall | +1.5494 pp |

Brier: R2 0.02116154; M2 0.02137020; delta +0.00020866.
Total pixel count differences (candidate minus control): {'tp': 26705, 'fp': 55995, 'fn': -26705}. Counts do not replace per-image macro metrics.

Final inspection 2026-09-14T05:39:59.829221+10:00; 751.236 seconds after training ended; inspections 2/2.

| Observed Train epoch | Residual RMS / feature RMS | Attention entropy | Q gradient absmax |
|---|---:|---:|---:|
| 1 | 0.000000 | 2.819544 | 0 |
| 10 | 0.128543 | 2.778918 | 0.00315673 |
| 40 | 0.132982 | 2.659946 | 0.00314849 |
| 80 | 0.118247 | 2.649230 | 0.00325572 |

These observations cover one ordinary Train batch at each of four epochs. Nonzero residuals/gradients show that the branch is active on those batches; they do not establish a beneficial causal effect or dataset-wide grounding.
