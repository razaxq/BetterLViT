# P8 RACE-Fuse V1 lock

- experiment: `p8_race_fuse_v1`
- purpose: validation-only candidate paired with C3
- epochs: 80
- physical batch size: 16
- seed: 1219
- text encoder: frozen CXR-BERT; LoRA disabled
- decoder: FAM-EPPA V4-B
- objective: Dice/Focal 0.5/0.5 plus RACE auxiliary weight 0.05
- boundary loss: 0
- RACE-Fuse: V1, four cross-scale routes, maximum residual strength 0.15
- Test split: prohibited
