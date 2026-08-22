# BetterLViT controlled paper ablation

This branch deliberately replaces the open-ended V4 experiments with five
pre-registered runs. All training is performed on the local AMD Radeon RX 7900
XTX with the same dataset split, seed, optimizer, augmentation, 200-epoch
budget, physical batch size 8 and early-stopping rule. Batch 8 replaces the
initial batch-16 registration after that configuration crashed in the Windows
ROCm/MIOpen BatchNorm kernel before producing a complete first checkpoint. A
real-data benchmark with MIOpen disabled measured about 30.9 images/second at
batch 8 versus 22.5 images/second at batch 4 (+37%), with about 5.03 GB of
dedicated GPU memory and no numerical or ROCm errors over more than 200 batches.

| ID | Profile | Text | Decoder | Objective | Role |
|---|---|---|---|---|---|
| B0 | `b0_baseline` | frozen CXR-BERT | original PLAM | Dice/BCE | thesis baseline |
| A0 | `a0_lora` | CXR-BERT + LoRA | original PLAM | Dice/BCE | LoRA-only anchor |
| A1 | `a1_lora_focal` | CXR-BERT + LoRA | original PLAM | Dice/Focal | LoRA + Focal |
| A2 | `a2_lora_freq` | CXR-BERT + LoRA | FAM-EPPA V4-B frequency path | Dice/BCE | LoRA + frequency |
| A3 | `a3_lora_fmiseg` | CXR-BERT + LoRA | FMISeg-adapted fusion | Dice/BCE | LoRA + FMISeg adaptation |

A3 is an adaptation of the interaction principles described by FMISeg, not a
claim to reproduce or rename its published dual-ConvNeXt architecture. It adds
bidirectional low/high-frequency interaction and bidirectional visual/text
interaction to BetterLViT decoder skips.

## Locked protocol

- Dataset: QaTa-COV19-v2, fixed split of 5716 train, 1429 validation and 2113
  test images.
- Seed: 1219.
- Epoch budget: 200; physical batch size: 8; the same configured
  early-stopping rule applies to all.
- Local Windows ROCm runs set `MIOPEN_FIND_MODE=FAST`, using FindDb or the
  immediate fallback instead of benchmarking all available solvers at every
  new process start.
- The backend deterministic restriction is disabled because the deterministic
  Windows MIOpen BatchNorm path stalled or crashed before a first checkpoint.
  `cudnn.benchmark` remains disabled, the seed remains 1219, and this runtime
  setting is identical for B0 through A3.
- MIOpen itself is disabled for the local paper runs after both its BatchNorm
  access violation and repeated six-minute first-batch stalls. PyTorch's HIP
  fallback kernels keep the model and objective unchanged; the identical
  backend setting is applied to every profile.
- Boundary loss: disabled for every profile.
- Primary result: test Dice and IoU at threshold 0.5.
- Secondary result: select one threshold using validation only, freeze it, then
  evaluate the test set once.
- The test set must not be used for checkpoint, architecture, loss or threshold
  selection.
- B0 through A3 must finish before considering any combination such as
  LoRA + FMISeg + Focal.

## Local commands

Run the synthetic forward/backward check before every full experiment:

```powershell
D:\Project\BetterLViT\.venv\Scripts\python.exe tools\smoke_paper_profile.py --experiment b0_baseline --batch-size 8
```

Start one local background training run (the launcher refuses a duplicate):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_paper_experiment.ps1 -Experiment b0_baseline
```

After choosing the best checkpoint using validation Dice, evaluate it with the
matching explicit profile:

```powershell
D:\Project\BetterLViT\.venv\Scripts\python.exe tools\evaluate_experiment.py --experiment b0_baseline --checkpoint <checkpoint>
```
