# M1 / M2 visual-value regrouping discovery

This continues the user's authorized search for a second structural innovation
complementary to EPPA. It follows the closed T1/T2 controls and frozen T2
interventions. The previous phrase/null-match entry condition did not pass.

Read PROTOCOL.md for the new, independently registered hypothesis and all gates.
The implementation has matching visual/text anchor variants and image-only
values. Neither variant is a confirmed innovation. No new Test access is allowed.

Frozen source SHAs, tags and paths: sources.json. Source deployment and GitHub
resolution: *_deployment.json and github_sources_verified.json. Preflight proof:
preflight_verified.json when present. Actual launch: *_launch.json and *_state.json
when present; absence means not dispatched. Never infer completion from a healthy
preflight or an early epoch. Completed metrics require *_results/validation.json,
independent_verification.json and completed runtime provenance.

Execution order: M1 then M2, 80 epochs each. After M1 completes, archive_completed,
analyze, upload_completed, verify_downloads, publish --completed-label m1;
then launch m2 regardless of M1's point estimate. Each script takes --label.
First health inspection about15 minutes after dispatch, then one predicted final
inspection. Use inspect_run.py and its local state guards; do not add manual
training status queries. Four ordinary Train batches record branch activity.

After M2 is fully verified and archived, assess_discovery.py applies the frozen
gates. A discovery pass only permits further seeds and mechanism ablations.
The final research objective remains unfulfilled until supporting evidence and
prior-art differentiation are obtained.

Storage cleanup is separate from evidence certification: backup_historical.py
preserves historical completed checkpoints without claiming fresh evaluations.
storage_cleanup.py requires fresh matching cloud hashes, complete Last metadata,
a retained sibling Best, and no active GPU/training/file references.
