# -*- coding: utf-8 -*-
"""Post-hoc diagnostic for FreqEPPA learnable parameters.

Loads the best checkpoint of `betterlvit.config.test_session` and inspects each
decoder stage (up1..up4)'s EPPA submodule on CPU.  No GPU, no dataset, no
training -- runs in seconds.

Set `betterlvit/config.py:test_session` to the run dir holding the checkpoint
to inspect (same field that `scripts/test.py` consumes), then:

    python scripts/diagnose_eppa.py

Output answers three questions for each stage:
  1. tau_scale per-channel distribution (pos/neg/~zero count, abs_mean,
     abs_max, std, text histogram).
  2. zero-init parameters (ch_mlp[2].weight, text_proj.{weight,bias},
     sp_proj.weight) -- did Adam move them off zero?
  3. low_pass kernel -- has it stayed predominantly low-pass (per-channel
     sum still ~ 1.0, the Gaussian init value)?
"""
import os

# Force CPU; we only need to inspect parameter tensors, not run inference.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from betterlvit import config


def text_hist(t, bins=11, width=40):
    """Print a horizontal histogram of a 1-D tensor on stdout."""
    t = t.flatten()
    lo, hi = float(t.min()), float(t.max())
    if hi == lo:
        hi = lo + 1e-9
    edges = torch.linspace(lo, hi, bins + 1)
    counts = torch.histc(t, bins=bins, min=lo, max=hi)
    cmax = max(int(counts.max().item()), 1)
    for i in range(bins):
        bar = '#' * int(width * counts[i].item() / cmax)
        print(f"  [{edges[i]:+.4e}, {edges[i+1]:+.4e}]  {int(counts[i].item()):>4}  {bar}")


def main():
    if not config.test_session:
        raise SystemExit(
            "config.test_session is empty. Set it in betterlvit/config.py to the "
            "run dir holding the checkpoint to inspect (e.g. "
            "'Test_session_05.08_15h22'), same value scripts/test.py consumes."
        )

    ckpt_path = (
        f"./{config.task_name}/{config.model_name}/"
        f"{config.test_session}/models/best_model-{config.model_name}.pth.tar"
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']
    print(f"  keys in state_dict: {len(sd)}")
    print(f"  best_epoch: {ckpt.get('best_epoch', '?')}  "
          f"max_dice: {ckpt.get('max_dice', '?')}")

    for stage in ('up4', 'up3', 'up2', 'up1'):
        print(f"\n========== {stage}.eppa ==========")

        # 1. tau_scale distribution -------------------------------------------
        key = f"{stage}.eppa.tau_scale"
        if key in sd:
            t = sd[key].float()
            n_pos = int((t > 1e-6).sum().item())
            n_neg = int((t < -1e-6).sum().item())
            n_zero = int((t.abs() <= 1e-6).sum().item())
            print(f"  tau_scale: shape={tuple(t.shape)}  "
                  f"pos={n_pos}  neg={n_neg}  ~zero={n_zero}")
            print(f"             mean={float(t.mean()):+.4e}  "
                  f"abs_mean={float(t.abs().mean()):.4e}  "
                  f"abs_max={float(t.abs().max()):.4e}  "
                  f"std={float(t.std()):.4e}")
            text_hist(t)
        else:
            print(f"  tau_scale: <not in state_dict> (ckpt predates 2ab7c34?)")

        # 2. zero-init parameters: did Adam move them? ------------------------
        for sub in ('ch_mlp.2.weight', 'text_proj.weight', 'text_proj.bias',
                    'sp_proj.weight'):
            k = f"{stage}.eppa.{sub}"
            if k in sd:
                w = sd[k].float()
                print(f"  {sub:<22} shape={str(tuple(w.shape)):<14}  "
                      f"fro_norm={float(w.norm()):.4e}  "
                      f"abs_max={float(w.abs().max()):.4e}")
            else:
                print(f"  {sub:<22} <not in state_dict>")

        # 3. low_pass kernel: per-channel sum (init was 1.0) ------------------
        k = f"{stage}.eppa.low_pass.weight"
        if k in sd:
            w = sd[k].float()                          # [C, 1, 3, 3]
            sums = w.sum(dim=(1, 2, 3))                # [C]
            print(f"  low_pass.sum  per-ch  "
                  f"mean={float(sums.mean()):.4f}  "
                  f"std={float(sums.std()):.4f}  "
                  f"min={float(sums.min()):.4f}  "
                  f"max={float(sums.max()):.4f}  (init = 1.0)")
        else:
            print(f"  low_pass.weight: <not in state_dict>")


if __name__ == '__main__':
    main()
