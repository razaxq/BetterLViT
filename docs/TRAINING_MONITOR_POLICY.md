# Training monitoring policy

User instruction, 2026-09-07: at most two inspections per training run. Reserve
the final inspection until after training ends, and begin it no later than
30 minutes after training completion.

1. One progress inspection around the midpoint. Skip it if training has already
   finished. User-requested ad hoc checks are not silently repeated by timers.
2. Predict the finish time from the existing midpoint timing, and schedule one
   final inspection about 15 minutes later. Connect briefly only at that appointment.
   Report actual completion time and the gap; if training/testing is still pending,
   state that fact without inventing results. Do not add a third inspection.

Latest user clarification: use prediction, not a continuously connected wait.
No persistent SSH session, completion listener or polling between inspections.
Stop the scheduled task after its one final invocation. Prediction error, early
failure, sleep or network outages can invalidate the desired completion window;
report that honestly rather than claiming an unconditional 30-minute guarantee.

For current P11, the setup-time inspection at epoch 40/80 is inspection 1. Only
the final inspection remains. Its source is frozen at
`2fc6ab5c8e4662d741fd8b994e55b780391948ac`; live output is
`/root/dual_grain_runs/p11_20260907`. Training began at 17:00:47 Sydney time;
epoch 40 was nearly complete at 19:27. Predicted finish is approximately 21:54,
with the sole remaining inspection at 22:10 on 2026-09-07 (Australia/Sydney).
The initial persistent listener was stopped and its automation was replaced.
`tools/wait_training_completion.py` and its test remain archived history only;
they are no longer used for P11 monitoring. The frozen training source is unchanged.

This controls inspections, not the 80-epoch optimization or automatic Test
evaluation. A single-seed Test result is not proof of stable gain. For local
Codex scheduling, keep the computer awake, Codex running and SSH connectivity
available; an app or network outage can delay the final inspection beyond the
target. Do not represent local scheduling as an unconditional delivery guarantee.
