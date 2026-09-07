# Storage and source preservation — 2026-09-07

## Verified completed actions

All current local experiment branches/tags were pushed additively to
https://github.com/razaxq/BetterLViT. No main-branch merge or force push was used.
Previously uncommitted historical sources were preserved separately:

| Branch | Commit | Scope |
|---|---|---|
| archive/grouped-protocol-source-20260907 | cb17bea08cd56f3c78c0c95d8b0c56e5e3541399 | Historical grouped protocol draft |
| archive/4090d-runtime-source-20260907 | 6820693a14c22df5d56d933efdeb98243e0a13cb | Historical runtime and provenance changes |
| archive/server-tcsrv2-draft-20260907 | 5995632ff42cfb8b45e631d03cd732abb8f783d5 | Historical server TCSR V2 draft |

These source archives are not newly validated experiments. Original draft
formatting was retained. Dataset links, credentials and binary weights are excluded.

The training disk had 3,469,713,408 free bytes before cleanup and
12,893,089,792 free bytes afterward. Exactly 11 historical Last checkpoints
(9,423,352,281 logical bytes) were deleted after BOTH their own HF copy and
the sibling Best HF copy matched the source size and Xet hash. All sibling
Best checkpoints were preserved. Process/open-file checks and unchanged-file
identity checks preceded deletion. See `cleanup_completed.json`.

Shared storage remained 18,559,782,256 apparent bytes, below its explicit
20,000,000,000-byte hard limit. The local C: and D: disks had ample space.

## Current-result backup

Completed: all four prefixes and all 20 files were re-read from the Bucket and
matched source byte sizes and Xet hashes, totaling 6,800,659,149 logical bytes.
See `new_runs_upload_verified.json`. The successful server fallback used academic
acceleration, the isolated hf_xet 1.6.0 client, fixed upload concurrency 1, disabled
global dedup queries, and the API's raw-bytes upload path. These settings were
changed together, so this does not isolate which change resolved the stalls.
The redundant local relay/prefetch was stopped after HF verification; a verified
local C4 Best copy is retained. The 23-worktree publication audit is recorded in
`published_sources_verified.json`.

`transfer_manifest.json` records the exact 20 source artifacts for completed
80-epoch C4/P9/C8/P10 exploratory experiments: four sets of session log, Test
evaluation JSON, Best, Last, and TensorBoard event. Full source commit, checkpoint
metadata, source byte sizes and Xet hashes were checked before transfer.
Backup completion must be established by `new_runs_upload_verified.json`
containing all four arms and 20 verified remote files, not by the transfer manifest.

AutoDL academic acceleration was enabled for server uploads. Bulk/adaptive
uploads were interrupted after extended inactivity; a local relay and bounded
server concurrency are fallback attempts. These are operational observations,
not a diagnosed Xet defect. Uploads are additive; no remote files are deleted.

Scripts contain historical absolute paths and require adaptation before reuse.
Credentials are read from the environment/cache or stdin, never saved here.
Model downloads, partial files, Xet caches, raw network logs and Git bundles
are deliberately outside this source archive.
