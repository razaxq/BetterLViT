# P11 continuation from epoch 80 to 150

User requested continuation after reviewing the 80-epoch Test result. Resume the
original Last checkpoint, add 70 epochs, retain Adam moments, the unchanged
10-epoch cosine warm-restart schedule, sampler/worker/global random states,
and the first 80 validation-history rows. No architecture, loss, augmentation,
batch size, seed or threshold changes. Original P11 source is
`2fc6ab5c8e4662d741fd8b994e55b780391948ac`.

The resume loader now loads on CPU so CPU random-state ByteTensors remain valid;
model and optimizer loaders transfer their own state to CUDA. Original 80-epoch
artifacts remain untouched in a separate checkout. The continuation runner
records source SHA, parent SHA, parent checkpoint hashes and runtime versions.

Select the best validation-IoU checkpoint across all 150 epochs; if epoch 80
remains best, evaluate the untouched original Best using the original evaluator
checkout. Export validation and then explicitly authorized Test at threshold 0.5.
Report P11 80 versus continued 150 descriptively. C4 at 80 epochs is only a
historical reference, not a matched-budget architectural ablation. This is an
exploratory continuation after Test access, not a preregistered fresh 150-epoch run.

Launch with `tools/run_p11_continuation.py --parent-run <original-run> --output <new-run>`
in an isolated checkout with the dataset link and original runtime available.
No output directory or training session may already exist. At least 5 GiB free
space and an exclusive experiment lock are required.

Monitoring uses prediction and brief scheduled SSH snapshots. The launch sanity
check counts as inspection 1 of 2 for this continuation. Schedule only the final
inspection approximately 15 minutes after predicted completion, aiming for no
more than 30 minutes after actual training ends. Never keep SSH connected, poll,
or add an automatic third check. Pause the timer after the final inspection.
