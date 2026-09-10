The first probe stopped before model/data loading at the source-cleanliness assertion.
All three repositories report only `?? datasets`; this is the existing dataset symlink.
The revised check permits exactly this untracked symlink and rejects every other change.
No model forward, gradient calculation, or weight update occurred in that attempt.
