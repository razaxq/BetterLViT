# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

BetterLViT is a USYD Dissertation-A research fork of [LViT](https://arxiv.org/abs/2206.14718) (Language-meets-Vision
Transformer) for medical image segmentation on QaTa-Covid19 and MoNuSeg. The dissertation goal is to improve Dice / mIoU
on top of vanilla LViT through targeted architectural innovations (currently: CXR-BERT text encoder + LoRA, EPPA
edge-preserving attention, boundary-aware composite loss).

The local Windows checkout has **no Python runtime** — training/testing runs on a remote GPU box and the repo is synced
via git. Verify changes by reading code, not by executing scripts. See `memory/MEMORY.md` for collaboration
preferences (Chinese-language dialogue, ask-before-defaulting on architecture choices).

## Commands

All commands assume the remote GPU environment (Python 3.11, CUDA 12.8, RTX 5090). They will fail locally.

```
# Install (CUDA 12.8 wheels for torch)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install -r requirements.txt

# Editable install of the betterlvit package (recommended) so scripts can
# import `betterlvit.*` without sys.path manipulation. Run once.
pip install -e .

# Train (task / model / hyperparams all driven by betterlvit/config.py — there is no CLI)
python scripts/train.py

# Test the best checkpoint of a session — set betterlvit/config.py: test_session first
python scripts/test.py
```

If you prefer not to install, `python -m scripts.train` from the project root works equivalently.

There are no tests, linter, or build step in this repo. Configuration is global module state in `betterlvit/config.py`;
switching dataset or model means editing that file and re-running.

## Configuration

`betterlvit/config.py` is the single source of truth. Key knobs:

- `task_name` — `'Covid19'` or `'MoNuSeg'`. Drives dataset paths and `learning_rate` (3e-4 vs 1e-3).
- `model_name` — `'BetterLViT'` (LoRA-tuned CXR-BERT) or `'LViT'` (frozen CXR-BERT baseline) or `'LViT_pretrain'` (loads
  UNet weights from a hardcoded MoNuSeg checkpoint).
- `text_encoder_name` — `microsoft/BiomedVLP-CXR-BERT-specialized` (HF hub). Domain-aligned BERT for chest X-ray
  reports.
- `text_max_len` — token sequence length (default 32). Must match `Vit.CTBN3.in_channels` — changing this requires a
  corresponding edit to LViT internals.
- `text_use_lora` + `text_lora_*` — LoRA config; only applied when `model_name == 'BetterLViT'`.
- `resume_path` — set to a `.pth.tar` to resume; a **new session folder** is still created so the source `best_model` is
  never overwritten. `resume_max_dice` is a fallback only for old checkpoints missing the `max_dice` field.
- `enable_bark`, `shutdown_after_train` — side-effect kill-switches (default `True` to mirror legacy behavior). Disable
  to skip Bark pushes / `os.system("shutdown")`.
- `seed`, `cudnn.deterministic` — `seed` lives in `betterlvit/config.py`; `cudnn.deterministic` and the
  `random/np/torch/torch.cuda` seed calls happen in `scripts/train.py` `__main__`.
  `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` and `PYTHONHASHSEED` are also set at the top of `scripts/train.py` /
  `scripts/test.py` (before any torch import) — moved out of `config.py` so the env vars take effect deterministically.
  Multi-GPU upsampling still has nondeterminism (see README §5).

Each run creates `<task_name>/<model_name>/Test_session_MM.DD_HHhMM/` containing `models/`, `tensorboard_logs/`,
`visualize_val/`, and a `.log` file.

## Architecture

The model pipeline assembles three independently-developed pieces. Understanding the data flow requires reading them
together:

```
ImageToImage2D (betterlvit/data/dataset.py)
  └─ tokenizes Description -> (input_ids, attention_mask) of length text_max_len
  └─ returns sample: {image, label, input_ids, attention_mask}
       │
       ▼
BetterLViT.forward(image, input_ids, attention_mask)        # betterlvit/models/better_lvit.py
  ├─ encode_text(...) -> [B, seq_len, 768]                  # HF AutoModel + optional PEFT LoRA
  └─ super().forward(image, text)
       │
       ▼
LViT.forward(image, text)                                   # betterlvit/models/lvit.py
  ├─ U-Net encoder/decoder with 4 down + 4 up stages
  ├─ text_module4..1 (Conv1d): 768 -> 512 -> 256 -> 128 -> 64
  ├─ VisionTransformer (betterlvit/models/vit.py) at each scale: cross-attends image patches with text_k
  ├─ UpblockAttention (betterlvit/models/lvit.py) on each skip: applies EPPA(skip, text) before concat
  └─ Sigmoid head -> [B, 1, 224, 224]
```

Key cross-cutting invariants (don't break these without a deliberate plan):

- **Text dim is hardwired to 768.** `LViT.text_module4` is `Conv1d(in=768, out=512)`. Any text encoder swap must produce
  768-d hidden states, or that conv must be edited.
- **Sequence length flows through `text_seq_len`** from `config.text_max_len` -> `BetterLViT.__init__` ->
  `LViT.__init__` -> `VisionTransformer(text_seq_len=...)` -> `Vit.CTBN3` (a `Conv1d(in_channels=text_seq_len, ...)`).
  Changing `text_max_len` without re-threading this will silently break shape alignment in CTBN3.
- **EPPA uses the [CLS] token** (`text[:, 0, :]`) for channel-attention conditioning, by design.
  CXR-BERT-specialized's [CLS] is contrastive-aligned and free of [PAD] pollution. Do not switch to mean-pool without
  understanding the [PAD] contamination tradeoff.
- **Loss is `WeightedDiceBCE`** (Dice + BCE, both weight 0.5), assembled in `scripts/train.py`. It returns a single
  scalar; `betterlvit/engine/train_loop.py` does not expect a tuple. The boundary-aware `WeightedTverskyFocalBoundary`
  from earlier prototypes is no longer wired in (kept around in legacy git history if needed for ablation).
- **Sigmoid is inside the model** (`LViT.last_activation`). All losses operate on probabilities in [0, 1], not logits —
  using BCEWithLogitsLoss would double-sigmoid.
- **`val_loader` uses `shuffle=True`** (LViT upstream legacy). Changing it shifts CPU RNG consumption and breaks
  bit-exact reproduction of pre-refactor runs. Treat as a separate, baseline-rerun-required change.

## Innovations on top of vanilla LViT

Each is documented inline in its own file. When extending, prefer adding a new module to swapping these in place — they
are dissertation experiments and need to remain comparable.

1. **CXR-BERT text encoder + LoRA** (`betterlvit/models/better_lvit.py`): replaces legacy `bert_embedding` with
   `microsoft/BiomedVLP-CXR-BERT-specialized`. LoRA targets configurable via `config.text_lora_target_modules` (defaults
   to query/key/value/intermediate.dense/output.dense — all 6 BERT linears). `model_name='LViT'` toggles to a
   fully-frozen baseline for ablation.
2. **EPPA — Frequency-Routed Edge-Preserving Attention** (`betterlvit/models/eppa.py`): drop-in replacement for
   `PixLevelModule` on `UpblockAttention` skip connections. Decomposes the skip via a learnable depthwise 3×3 low-pass (
   init as Gaussian [1,2,1;2,4,2;1,2,1]/16) into `x_low` and `x_high = x - x_low`. Routes them: low-frequency drives *
   *channel attention** (`ca = 1 + 0.5·tanh(...)` ∈ (0.5, 1.5), conditioned on text [CLS] added pre-tanh);
   high-frequency drives **spatial attention** on `|x_high|` (`sa = 1 + tanh(...)` ∈ (0, 2), allowing
   unsharp-masking-style edge sharpening when sa > 1). Recomposes as `x_low·ca + x_high·sa`. **Identity at init via
   zero-init of `ch_mlp[-1]`, `text_proj`, and `sp_proj`** (last-layer-zero is the DiT/ControlNet trick — own gradients
   are non-zero at init, earlier layers move once the last layer leaves zero). No LayerScale gate — the previous
   CBAM-style EPPA's gate was diagnosed to die under Adam pressure (mean negative, abs_mean shrinking) because
   `x_sa = x · ca · sa` was structurally bounded by `x` and could not inject boundary information beyond what the skip
   already contained; FreqEPPA breaks the bound via `x_high · sa` where sa can exceed 1. **Train from scratch only** (
   state_dict shape differs from PLAM and prior EPPA).
3. **Loss library** (`betterlvit/losses.py`): currently active loss is `WeightedDiceBCE(0.5, 0.5)`. Other classes (
   Tversky/Focal/Boundary variants, multi-class Dice) are kept verbatim from legacy `utils.py` for ablation reuse but
   not imported in the live training path.

## Resume / checkpointing

`save_checkpoint` (in `scripts/train.py`) writes both `best_model-{model}.pth.tar` (only on dice improvement, after
epoch 5) and `last_model-{model}.pth.tar` (rolling, every epoch). The state dict carries `epoch_history` so resumed runs
continue printing the full per-epoch table. The scheduler is stepped **inside the val pass** (in
`betterlvit/engine/train_loop.py`), so when resuming with an old-format checkpoint that lacks `lr_scheduler` state,
`lr_scheduler.step(start_epoch)` correctly fast-forwards.

`thop` (FLOPs counter) is intentionally not used — it double-registers hooks on PEFT-wrapped modules and breaks training
after `.cuda()`. Param counts only.

## Notifications

`scripts/train.py` posts to a hardcoded Bark key (`uAnJRvt7pxbzE9KK6bCVva`, in `betterlvit/notify.py`) on training
start, on each new best Dice, and on training end, gated on `config.enable_bark`. Failures are swallowed (3s timeout).
The script also runs `os.system("shutdown")` after training completes when `config.shutdown_after_train` is true (
default) — be aware before kicking off a quick experiment, or set the flag to `False` in `betterlvit/config.py`.

## Repository layout

```
betterlvit/                 # library code (importable package)
  config.py                 # global config (was Config.py)
  losses.py / metrics.py / schedulers.py / io.py   # split from legacy utils.py
  notify.py                 # Bark client
  data/dataset.py           # was Load_Dataset.py
  engine/train_loop.py      # was Train_one_epoch.py
  models/{better_lvit,lvit,vit,unet,eppa,pixlevel}.py   # was nets/*.py
scripts/
  train.py / test.py        # entry scripts (were train_model.py / test_model.py)
pyproject.toml              # editable install via `pip install -e .`
```
