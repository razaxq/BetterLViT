# BetterLViT controlled paper ablation

This branch deliberately replaces the open-ended V4 experiments with five
pre-registered runs. The current canonical protocol runs on one RTX 4090D
server with the same dataset split, seed, optimizer, augmentation, 150-epoch
budget, physical batch size 16 and early-stopping rule. Training uses
`drop_last=True`, so every optimizer step keeps the same physical shape and four
shuffled samples (0.07%) are omitted per epoch. The same rule is locked for all
formal profiles.

| ID | Profile | Text | Decoder | Objective | Role |
|---|---|---|---|---|---|
| B0 | `b0_baseline` | frozen CXR-BERT | original PLAM | Dice/BCE | thesis baseline |
| A0 | `a0_lora` | CXR-BERT + LoRA | original PLAM | Dice/BCE | LoRA-only anchor |
| A1 | `a1_lora_focal` | CXR-BERT + LoRA | original PLAM | Dice/Focal | LoRA + Focal |
| A2 | `a2_lora_freq` | CXR-BERT + LoRA | FAM-EPPA V4-B frequency path | Dice/BCE | LoRA + frequency |
| A3 | `a3_lora_fmiseg` | CXR-BERT + LoRA | FMISeg-adapted fusion | Dice/BCE | LoRA + FMISeg adaptation |
| A4 | `a4_lora_freq_focal` | CXR-BERT + LoRA | FAM-EPPA V4-B frequency path | Dice/Focal | A2 with Focal only |

A3 is an adaptation of the interaction principles described by FMISeg, not a
claim to reproduce or rename its published dual-ConvNeXt architecture. It adds
bidirectional low/high-frequency interaction and bidirectional visual/text
interaction to BetterLViT decoder skips.

## Locked protocol

- Dataset: QaTa-COV19-v2, fixed split of 5716 train, 1429 validation and 2113
  test images.
- Seed: 1219.
- Epoch budget: 150; physical batch size: 16; training `drop_last=True`; the same configured
  early-stopping rule applies to all.
- Runtime: one RTX 4090D, CUDA/cuDNN enabled, deterministic algorithms required,
  cuDNN benchmarking and TF32 disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Dataset ordering, sampler state, worker seeds and global CPU/CUDA RNG states
  are stored so an epoch-boundary resume preserves the training trajectory.
- Boundary loss: disabled for every profile.
- Primary result: test Dice and IoU at threshold 0.5.
- Secondary result: select one threshold using validation only, freeze it, then
  evaluate the test set once.
- Report both macro (mean per image, primary) and micro (global pixels) Dice/IoU
  for validation and test.
- The test set must not be used for checkpoint, architecture, loss or threshold
  selection.
- B0 through A3 must finish before considering a combination such as
  LoRA + FMISeg + Focal.

## Server commands

Run the synthetic forward/backward check before every full experiment:

```bash
/root/autodl-tmp/envs/betterlvit-paper/bin/python \
  tools/smoke_paper_profile.py --experiment b0_baseline --batch-size 16
```

Start one local background training run (the launcher refuses a duplicate):

```bash
bash scripts/start_paper_experiment_server.sh b0_baseline 1219 150 16
```

After choosing the best checkpoint using validation Dice, evaluate it with the
matching explicit profile:

```bash
/root/autodl-tmp/envs/betterlvit-paper/bin/python \
  tools/evaluate_experiment.py --experiment b0_baseline --checkpoint <checkpoint>
```
