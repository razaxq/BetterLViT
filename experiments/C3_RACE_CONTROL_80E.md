# C3 RACE control lock

- experiment: `c3_race_control`
- purpose: validation-only control for P8 RACE-Fuse V1
- epochs: 80
- physical batch size: 16
- seed: 1219
- text encoder: frozen CXR-BERT; LoRA disabled
- decoder: FAM-EPPA V4-B
- objective: Dice/Focal 0.5/0.5
- boundary loss: 0
- RACE-Fuse: disabled
- Test split: prohibited
