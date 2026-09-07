# Training monitoring policy

User instruction, 2026-09-07: at most two inspections per training run. Reserve
the final inspection until after training ends, and begin it no later than
30 minutes after training completion.

1. One progress inspection around the midpoint. Skip it if training has already
   finished. User-requested ad hoc checks are not silently repeated by timers.
2. One final inspection, triggered by completion. For the Train -> Val -> Test
   runner, wait for the result chain to complete, with a maximum 20-minute grace
   after `training.status` is written. If evaluation is still pending, report that
   fact without inventing metrics. Training failures trigger the final inspection
   immediately. Do not launch a third scheduled inspection.

Between these inspections, an operating-system event wait may listen for the
status file or runner exit. It does not query training logs, checkpoints or the
GPU. Do not replace this with repeated model inspections or repeated status
reports. Stop the scheduled task once the final report has been delivered.

For current P11, the setup-time inspection at epoch 40/80 is inspection 1. Only
the final inspection remains. Its source is frozen at
`2fc6ab5c8e4662d741fd8b994e55b780391948ac`; live output is
`/root/dual_grain_runs/p11_20260907`. `tools/wait_training_completion.py` supports
the runner's existing status files without modifying the frozen training source.

This controls inspections, not the 80-epoch optimization or automatic Test
evaluation. A single-seed Test result is not proof of stable gain. For local
Codex scheduling, keep the computer awake, Codex running and SSH connectivity
available; an app or network outage can delay the final inspection beyond the
target. Do not represent local scheduling as an unconditional delivery guarantee.
