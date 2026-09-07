# P11 continuation launch, 2026-09-07

User authorized continuing P11 from completed epoch 80 to total epoch 150.
Frozen continuation source: `c724a62001f6c3b809cb12eee78f3bee79bdb60a`.
Tag: `paper-p11-150e-resume-b16-seed1219-20260907`.
Branch: `paper/p11-150e-resume`; original 80-epoch checkout remains unchanged.

The real resume loader was exercised twice on CUDA with the original Last
checkpoint, real shuffled Train loader (batch 16, four workers) and one optimizer
step. Both runs restored all model tensors, Adam states, scheduler, Python/NumPy/
CPU/CUDA/sampler random states and 80 history rows exactly. Sample names, loss,
prediction hash and updated trainable-weight hash matched in both processes.
No Test data or checkpoint writes occurred during this preflight.
Preflight source was `d7810def636acf61b68d26708a893b36d10be2a6`; the only subsequent
change before freezing was a comparison regression test. That test passed the
new-best, retained-best, wrong-threshold and wrong-case scenarios locally.

Continuation uses the same architecture, loss, seed, batch size and learning-rate
schedule. It restores epoch 80's Last and trains epochs 81 through 150. Best is
selected across the complete validation history. If epoch 80 remains best, the
original Best and its original evaluator source are retained. Automatic validation
and Test exports use fixed threshold 0.5. The result is an exploratory extension
requested after viewing Test; the 80-epoch C4 is not an equal-budget ablation.

Remote run: `/root/dual_grain_runs/p11_150_20260907`.
Expected duration is approximately 4 hours 18 minutes, based on the prior
80-epoch run taking 4 hours 54 minutes. The launch sanity snapshot counts as
inspection 1 of 2; the sole final inspection is scheduled for 03:00 Sydney on
2026-09-08, around 15 minutes after predicted completion. Prediction error is
possible. Never poll or keep SSH connected; record actual end/inspection times
at the appointment and pause the timer after the final snapshot.

Adjacent JSONs contain the exact runtime, checkpoint hashes, launch state and
repeated preflight evidence. Training results are not yet available in this archive.
