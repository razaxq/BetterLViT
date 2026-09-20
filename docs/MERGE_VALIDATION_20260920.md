# Final implementation integration validation

Integration parent: `561b8fa9489b71d9c3c222b7f4e70971c8556116`.
Scope: existing final FSDR + complete RACE implementation, compatible FSDR naming,
paper reports and original result provenance. No training or Test inference was run.

Passed locally using Python 3.10 / PyTorch 2.5.1 CPU:

- `tools/check_stage1.py`: historical profiles, invalid manifests, exporter CLI,
  unchanged metric loop and helpers.
- `tools/check_race_fuse.py`: routing identity initialization and determinism.
- `tools/check_p8_restart.py`: binding semantics, 24 count/fallback cases and profile.
- `tools/check_cos_restart.py`: all 80 actual scheduler rates, profile invariance,
  manifest rejection, exporter CLI and unchanged metric calculation. Used the
  existing cr1s1219 manifest as a temporary fixture; it is not a new active run.
- Existing FSDR Haar, adaptive forward/backward and nonadaptive-stage checks;
  `nets.fsdr.FSDR is nets.eppa.EPPA`; LViT import and final decoder configuration.
  The legacy whole-script configuration assertion requires an obsolete bare
  architecture ID, so its structural checks were called directly with the final
  `p8_r2_binding` configuration instead of changing historical profile IDs.
- Archived result snapshot SHA-256 verification, JSON parsing, changed Python
  syntax and credential-pattern scan of imported results and tracker.

These are CPU integration checks, not a new GPU training reproduction. Original
training and evaluation evidence remains attributed to the recorded runtime SHAs.
