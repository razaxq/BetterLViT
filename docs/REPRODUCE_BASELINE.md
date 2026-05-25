# Reproducing the LViT-T QaTa-COV19 baseline

Branch: **`repro/lvit-baseline-bertbase`** (forked from upstream `a9b0b41`).

**Goal:** reproduce the official LViT-T score on QaTa-COV19 — **Dice 83.66 / IoU 75.11** —
*before* making any model changes, so later improvements (CXR-BERT, LoRA, EPPA) can be measured
as single-variable experiments against a trusted baseline.

## What this branch changes (and what it does not)

The only thing that ever broke on the RTX 5090 was the unmaintained mxnet `bert-embedding`
package — **not** the `bert-base-uncased` weights. This branch swaps just that:

- Text features are now produced by **frozen HuggingFace `bert-base-uncased`**, **precomputed
  offline** and cached to disk (`tools/precompute_text_emb.py`), then read by `utils.read_text_emb()`.
- **Unchanged:** the LViT-T architecture (PLAM), the loss (`WeightedDiceBCE` 0.5/0.5), the
  optimiser/scheduler (Adam, `lr=3e-4`, cosine `T_0=10, eta_min=1e-4`, **no weight decay**), and
  the model's `(B, 10, 768)` text input. **No** CXR-BERT, **no** LoRA, **no** EPPA.

## Prerequisites

- An NVIDIA RTX 5090 (or any CUDA-12 GPU) instance — e.g. an AutoDL rented server.
- The QaTa-COV19 dataset laid out under `datasets/Covid19/` with the text spreadsheets:
  - `datasets/Covid19/Train_Folder/Train_Val_text.xlsx` (+ `img/`, `labelcol/`)
  - `datasets/Covid19/Val_Folder/`   (`img/`, `labelcol/`)
  - `datasets/Covid19/Test_Folder/Test_text.xlsx` (+ `img/`, `labelcol/`)

## Steps

### 0. Switch to the branch
```bash
git switch repro/lvit-baseline-bertbase
```

### 1. Install the environment (CUDA 12.8 wheels)
```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt
```
Quick sanity check that the 5090 issue is gone:
```bash
python -c "import torch, transformers; print(torch.cuda.is_available()); \
from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased'); print('ok')"
```

### 2. Precompute the frozen text embeddings (run once)
```bash
python tools/precompute_text_emb.py \
    --xlsx datasets/Covid19/Train_Folder/Train_Val_text.xlsx \
    --xlsx datasets/Covid19/Test_Folder/Test_text.xlsx
```
This writes sibling caches `Train_Val_text.emb10.pt` and `Test_text.emb10.pt`
(`{image_key: float32 ndarray (10, 768)}`). Useful flags:
- `--max-len 10` — fixed token length (default 10, the original QaTa cap).
- `--add-special-tokens` — OFF by default to mirror the legacy `bert_embedding`; turn on only
  to A/B if the reproduction comes out low.

### 3. Train
```bash
python train_model.py
```
Key settings live in `Config.py`: `task_name='Covid19'`, `model_name='LViT'`, `learning_rate=3e-4`,
`batch_size=4`, `epochs=2000` with early stopping.

### 4. Test
Copy the trained `session_name` into `Config.py` (`test_session = "..."`), then:
```bash
python test_model.py
```

## Success criterion

Test-set **Dice ≈ 0.8366 / IoU ≈ 0.7511** (within run-to-run noise). If it lands there, tag the
run and treat this branch as the canonical baseline that future experiments re-fork from. If it
lands materially low, the search space is small (architecture/weights are now identical to
official): bisect `batch_size`, `epochs`, and the `--add-special-tokens` knob.

## Open items to confirm

- **Official Covid19 `batch_size` / `epochs` / `early_stopping_patience`:** the upstream `Config.py`
  only ships MoNuSeg defaults; this branch assumes `batch=4`, `lr=3e-4`, no weight decay. Confirm
  against the official LViT repo's Covid19 config and adjust if needed.
- **`add_special_tokens`:** kept a one-line knob in the precompute script (default off).

## Notes

- If you stashed `.idea/` files when branching away from another branch, restore them with
  `git stash pop` after switching back.
- Documentation caveat for the dissertation: upstream LViT's text encoder is **bert-base-uncased**
  (768-d precomputed embeddings), **not** ClinicalBERT.
