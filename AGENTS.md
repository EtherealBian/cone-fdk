# Cone-FDK: working and retention rules

Develop, compile, test and run CUDA on DGX Spark. Windows is an SSH and migration endpoint, not a fallback compute node.

At the end of each iteration and after a successful handoff, inspect the **project's** scratch areas on both hosts. Remove obsolete outputs and reproducible caches once their producing job is finished. Remove superseded builds only after the replacement has passed its gates and there is a verified path back to any required historical build. A date, filename, extension or `git status` alone is not proof that something is disposable.

- Use explicit project paths. Resolve and verify the parent; reject symlinks/junctions in cleanup paths. Never sweep a drive, a home directory, unrelated repositories, the portfolio, system packages, global tool caches or SSH settings. Do not use blanket `git clean`.
- Keep raw scans, reference volumes, frozen baselines and receipts, uncommitted sources, and data or partial downloads still in use. On a failed run retain the smallest sufficient failure log/reproducer until diagnosed. Do not classify active locks, transfers, environments or tools as stale caches.
- For duplicate measurement outputs: first close the run, check for consumers, recompute the SHA-256 against both its receipt and a retained copy, then unlink only the verified extra files. Retain the signed/pinned provenance if present and keep the original receipt unchanged.
- Retire a Windows project tree only after checking the remote inventory and content checksums, a restore drill, and a final check that the source has not changed. If verification fails or ownership is ambiguous, retain it and report why.
- Record host, timestamp, exact path, reason, bytes unlinked, and retained replacement/receipt in the Spark migration workspace's `maintenance/` log. Logical bytes unlinked are not a guarantee about physical free space. Do not log secrets. Retire window-specific credentials only after the handoff is complete.

## Publishing qualified milestones

After a milestone passes its stated checks, publish its code and reviewable evidence to the project repository. Verify the remote branch tip after a non-forced push. Public summaries identify DGX Spark / GB10 only for runs measured there. Label historical timings by provenance, keep original machine receipts intact, and keep the portfolio separate. Never publish credentials, local host paths, raw scan archives, transient caches or private run logs. If a push fails, report the local commit and the error rather than calling it published.
