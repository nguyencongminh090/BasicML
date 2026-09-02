---
id: FIX-0015
timestamp: 2026-09-02T22:07:49+07:00
todo_id: TODO-0005, TODO-0006
---

## Prompt

Session started with "check for unfinished task". Found two completed feature
branches (`feature/TODO-0005-mlp-graph-demo`, `feature/TODO-0006-vanishing-gradient-demo`)
that were never merged into `dev` — each carrying one `BasicML/demo/*.py` script
plus its own TODO/fix-log trail, and neither TODO recorded in `dev`'s backlog.
User: "Merge two branch -> dev".

## Action

- `git merge --no-ff feature/TODO-0005-mlp-graph-demo` into `dev` (commit
  `baa4194`) — brings in `demo/plot_dynamic_mlp_graph.py`, `instructions/TODO-0005.md`,
  `todo/TODO-0005.md`, fix-log entries FIX-0006/0007/0010.
- `git merge --no-ff feature/TODO-0006-vanishing-gradient-demo` into `dev`
  (commit `f74ecf1`) — brings in `demo/plot_dynamic_vanishing_gradient.py`,
  `instructions/TODO-0006.md`, `todo/TODO-0006.md`, fix-log entries FIX-0008/0009.
- Both merges conflicted only in `ai-audit/todo/INDEX.md` and
  `ai-audit/fix-log/INDEX.md` (both sides appended rows since the merge-base
  `9792df9`). Resolved by keeping `dev`'s active `todo/INDEX.md` (just TODO-0007)
  and interleaving the incoming fix-log rows into chronological order
  (FIX-0006/0007 → FIX-0008/0009 → FIX-0010 → FIX-0011..0014).
- Closed both TODOs (commit `7dd4a4b`): `status: in-progress` → `done`,
  `git mv` `todo/TODO-0005.md` / `todo/TODO-0006.md` into `todo/archive/`, added
  their rows to `todo/archive/INDEX.md`.
- Deleted the two now-merged local feature branches.
- `pyrefly check`: the two new demo files add 0 errors (the 9 pre-existing
  errors are in `examples/check_gradients.py` / `train_linear.py`, unchanged and
  already present on `dev`).

## Decision

- `--no-ff` merges (not rebase/squash): the branches were already committed with
  their own history and audit trail; preserving the merge topology keeps the
  fix-log IDs traceable to their commits.
- The stale files on the branches (old `README.md`, deleted `LICENSE`, old
  `TODO-0007`) did **not** come across — those files were untouched by the branch
  commits, so the merge only pulled the demo + audit additions. No revert of
  FIX-0011..0014.
- Closed both TODOs as done rather than leaving them `in-progress`: the demos
  were feature-complete and pyrefly-clean on their branches; merging was the
  only remaining step.

## Conclusion

Done. `dev` now contains both demo scripts and a consistent, chronological
ai-audit trail. Feature branches deleted. No library code (`basicml/`) changed.

Not done (flagged to user, not in scope of this request): `README.md`'s
Interactive demos table does not yet list `plot_dynamic_mlp_graph.py` or
`plot_dynamic_vanishing_gradient.py` — a README refresh under TODO-0007.
