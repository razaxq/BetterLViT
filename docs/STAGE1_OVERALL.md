# Stage one: FSDR x whole binding-repaired RACE

Only eight runs are authorized: J0 and J2 seeds1219/2027/3407, J3 seeds2027/3407.
Existing FSDR-only three seeds and J3 seed1219 are reused. No route-only,
auxiliary-only or internal mechanism runs belong to this queue.

J0 is frozen CXR-BERT, original PLAM, no RACE, Dice/Focal. J2 uses original
PLAM and the full RACE route plus binding-repaired auxiliary objective. J3
retains historical p8_r2_binding exactly, including FSDR and full RACE.
All use no LoRA, 80 epochs, batch16,224px, Adam1e-4 weight decay, single cosine
3e-4 to1e-6, legacy augmentation, Dice/Focal0.5/0.5, no boundary loss.
RACE auxiliary weight0.05 and original0.4/0.4/0.2 subloss weights remain fixed.
Binding repair, uncovered-grammar legacy fallback, original count targets stay fixed.

New PLAM controls replay **fresh random FSDR decoder initialization**, copy only
common decoder convolution initialization, and preserve the subsequent RNG
state. Temporary FSDR blocks are discarded. This aligns shared CNN/ViT,
decoder convolutions, reconstruction/text layers and RACE weights with the
same-seed existing FSDR anchors, without copying any trained weights or
retaining FSDR operations in PLAM. Historical profiles take their unchanged
initialization path. Preflight checks every shared state tensor exactly.

Each run has its own manifest, full source commit and Git tag. Queue requires
completed Train-only preflight, uses immutable isolated worktrees, runs serially
and stops on failure. Training selects Best by Val macro IoU. The same Best
is exported on Val then Test at0.5 per standing authorization, with per-image
metrics, source checks and no Test tuning. A run is complete only after both
exports succeed. All preflight outputs use Train only, without optimizer steps.

Test has prior access. Report all paired seeds and mean/sample SD, not only
the best seed. Treat RACE as one module. No internal route-versus-supervision
claim is implied. Scientific success is assessed after complete results;
queue completion itself is not proof of positive gains.
