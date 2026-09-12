# T1 matched decoder control (validation only)

Source `72295aa38649fea8ffed2cdd7330c3a18504a330`; seed1219;80epochs;Best67.

| Model | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| R2 reused | 72.6654% | 82.4034% | 80.5086% | 88.0502% |
| T1 | 72.5589% | 82.2939% | 80.5614% | 87.8876% |

IoU delta -0.1064 pp; grouped descriptive CI [-0.3697382700878451, 0.15381280489653953] pp.

Single-seed discovery only; no Test access, stable gain or novelty claim. All small-area metrics and gradient observations are preserved.
Export-minus-checkpoint-selection IoU=-2.82794487827e-09. Training uses >=0.5 while frozen export uses >0.5; reconcile any material discrepancy before performance interpretation.

Smallest-mask quartile: 358 images, area <= 2075 pixels.
| Metric | Small-mask delta |
|---|---:|
| iou | -0.5918 pp |
| dice | -0.5197 pp |
| precision | -0.0758 pp |
| recall | -0.8321 pp |

Brier: R2 0.02116154; T1 0.02108096; delta -0.00008059.
Total pixel count differences (candidate minus control): {'tp': 2009, 'fp': 2616, 'fn': -2009}. Counts do not replace per-image macro metrics.

Final inspection 2026-09-13T06:58:11.863338+10:00; 625.488 seconds after training ended; inspections 2/2.

| Observed Train epoch | Residual RMS / feature RMS | Attention entropy | Q gradient absmax |
|---|---:|---:|---:|
| 1 | 0.000000 | 3.418351 | 0 |
| 10 | 0.075726 | 3.353207 | 0.00272401 |
| 40 | 0.093415 | 3.329891 | 0.00441111 |
| 80 | 0.094887 | 3.336595 | 0.00532505 |

These observations cover one ordinary Train batch at each of four epochs. Nonzero residuals/gradients show that the branch is active on those batches; they do not establish a beneficial causal effect or dataset-wide grounding.
