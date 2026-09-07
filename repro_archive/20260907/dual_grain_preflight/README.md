# P11 launch record — 2026-09-07

The user's requested Git preservation and storage maintenance were completed
before P11 implementation began. The HF completion proof covers 20 artifacts
(6,800,659,149 bytes) for C4/P9/C8/P10, with matching source/remote Xet hashes and
sizes. Storage cleanup freed 9,423,376,384 filesystem bytes while retaining all
sibling Best checkpoints. See the adjacent `maintenance` archive for full proof.

## Frozen P11 source

- Commit: `2fc6ab5c8e4662d741fd8b994e55b780391948ac`
- Branch: `paper/p11-dual-grain`
- Tag: `paper-p11-dual-grain-80e-b16-seed1219-20260907`
- Both branch and peeled tag were verified on GitHub before launch.
- Server: `/root/autodl-tmp/BetterLViT-dual-grain-p11`
- Run output: `/root/dual_grain_runs/p11_20260907`
- Start: 2026-09-07 07:00:47 UTC / 17:00:47 Australia/Sydney.

The 80-epoch experiment uses the C4 protocol with two 28x28 visual-query branches,
30,016 added parameters, and no new supervision or loss. It starts from scratch.
The main outcome is Test macro IoU at threshold 0.5; Best selection remains on
validation IoU. Training success triggers Val export, authorized Test evaluation,
and paired C4 comparisons automatically. This is an exploratory single-seed run.

## Completed checks and current status

Two independent CUDA preflights passed: common baseline weights/RNG and initial
predictions match exactly; fine positions differ once trained; coarse context is
used; all new parameters receive finite nonzero gradients after the second
backward pass. The real Train batch of 16 reproduced identical losses and output
SHA256 in both runs. Peak allocated/reserved memory was 15.7254/16.5566 GiB.
The two-step probe is not a training result and did not access Test.

Python parsing, Git whitespace checks, credential-pattern checks, evaluator and
runner CLI checks passed. Comparison sanity checks returned zero deltas/intervals
for identical predictions and rejected a mismatched training seed.

Launch was verified by the runtime SHA, separate runner and training processes,
GPU activity, and successful forward/backward updates in epoch 1. No completed
P11 training, validation result, Test result, or stable IoU gain is claimed here.
The authoritative live state is `chain.status`; `complete` requires every stage
to exit successfully. This document is a launch-time snapshot and can become stale.
