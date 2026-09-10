Initial source 618e32348869a8bc526a17b905579f1d992f3883 added broad LF rules that
made historical CRLF blobs appear modified on the Linux checkout. Deployment
and both requested prechecks stopped at source-cleanliness assertions before
model/data preflight. The follow-up removes broad rules; original model files
retain their historical contents. Updating this new, unlaunched development
worktree is allowed only after `git diff --ignore-space-at-eol --exit-code` and
an empty staged diff verify the change is solely line endings. Final sources
still require a fully clean tracked worktree.
