# RACE-PE V2: bounded repair check, 2026-09-07

V1 permitted an all-positive text solution and forced agreement between report
locations and image rectangles that do not share anatomical boundaries.
V2 predicts explicit mentions in encoder-visible text, with unmentioned locations
negative only for this language task. It removes report-to-mask consistency and
uses mention * visual presence * pixel extent for routing. V1 remains unchanged.

Before any new segmentation training, run semantic/gradient checks, the historical
V1 checks, and two deterministic batch-16 CUDA forward/backward preflights.
Then train only a randomly initialized text head over frozen CXR-BERT features.
Use seed 1219, AdamW lr 0.001, weight decay 0.0001, batch 128, fixed 40 epochs.
Hold out 20% of encoder-visible Train report templates. No Test data are read.
Do not select an epoch or tune parameters using the resulting scores.

Prospective language learnability gate: Val and held-out-template macro F1 >= 0.85,
Val macro specificity >= 0.90, Val F1 minus cyclic-shuffled F1 >= 0.15, and Val
BCE below the all-positive predictor. Passing establishes only language-head
learnability, not segmentation benefit or stable IoU gain. The diagnostic head
must not silently initialize later segmentation experiments.

Remaining risks include rectangle/anatomy mismatch during routing, unequal
auxiliary-loss effects, single-seed variance, and the already accessed historical
Test split. Further mechanism ablations must use Train/Val.
