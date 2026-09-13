# M1 matched decoder control (validation only)

Source `04be8e182d8802fb9e529e0a1f6d7cd2e2eb961d`; seed1219;80epochs;Best67.

| Model | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| R2 reused | 72.6654% | 82.4034% | 80.5086% | 88.0502% |
| M1 | 72.7827% | 82.5333% | 80.4997% | 88.3978% |

IoU delta +0.1173 pp; grouped descriptive CI [-0.1468756181991317, 0.3870853086534782] pp.

Single-seed discovery only; no Test access, stable gain or novelty claim. All small-area metrics and gradient observations are preserved.
Export-minus-checkpoint-selection IoU=1.79316983484e-09. Training uses >=0.5 while frozen export uses >0.5; reconcile any material discrepancy before performance interpretation.

Smallest-mask quartile: 358 images, area <= 2075 pixels.
| Metric | Small-mask delta |
|---|---:|
| iou | -0.0941 pp |
| dice | +0.0584 pp |
| precision | +0.2238 pp |
| recall | +0.2334 pp |

Brier: R2 0.02116154; M1 0.02103743; delta -0.00012412.
Total pixel count differences (candidate minus control): {'tp': 17816, 'fp': 10165, 'fn': -17816}. Counts do not replace per-image macro metrics.

Final inspection 2026-09-14T00:26:59.938150+10:00; 582.250 seconds after training ended; inspections 2/2.

| Observed Train epoch | Residual RMS / feature RMS | Attention entropy | Q gradient absmax |
|---|---:|---:|---:|
| 1 | 0.000000 | 3.418351 | 0 |
| 10 | 0.119241 | 3.344006 | 0.00143461 |
| 40 | 0.099470 | 3.159428 | 0.00277248 |
| 80 | 0.089781 | 3.132207 | 0.00273115 |

These observations cover one ordinary Train batch at each of four epochs. Nonzero residuals/gradients show that the branch is active on those batches; they do not establish a beneficial causal effect or dataset-wide grounding.
