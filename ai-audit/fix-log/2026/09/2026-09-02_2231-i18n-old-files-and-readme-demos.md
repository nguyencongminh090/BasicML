---
id: FIX-0016
timestamp: 2026-09-02T22:31:50+07:00
todo_id: TODO-0007
---

## Prompt

"1. You can update with two new demos. 2. Translate Vietnamese comment in old
files." (Continuation of the branch-merge session.) Follow-up
`AskUserQuestion`: scope for the 9 old `.py` files = **translate + add
docstrings** (bring them to the CLAUDE.md Code-style bar), not translate-only.

## Action

Branch `feature/TODO-0007-i18n-old-files` off `dev`.

- **README** (`## Interactive demos` table): added rows for
  `plot_dynamic_mlp_graph.py` and `plot_dynamic_vanishing_gradient.py`
  (merged in FIX-0015 but never listed); reworded the
  `plot_layer_transformations.py` row.
- **examples/** — translated the module docstrings from accentless Vietnamese
  and added Google-style docstrings to every function:
  `train_linear.py`, `train_logistic.py`, `check_gradients.py`.
- **demo/** — same treatment for `plot_dynamic_linear.py`,
  `plot_dynamic_logistic.py`, `plot_dynamic_3d_logistic.py`,
  `plot_dynamic_decision_boundary.py`, `plot_dynamic_layer_morphing.py`,
  `plot_layer_transformations.py`: module docstring, config-comment,
  `OneCycle` / `TrainingHistory` dataclass docstrings, per-function docstrings,
  and the nested `update`/`state_at` closures. Redundant what-comments dropped.
- **layer_space_transformation.ipynb** — translated all markdown cells (topology
  framing, section headings, the $Z_1 / H_1 / Z_2$ walkthrough), inline
  comments, plot titles, and the two stale Vietnamese cell-output strings.
- No code paths, parameters, or numeric config changed anywhere — this is a
  comment/docstring/markdown-only diff.

## Decision

- Accepted the "translate + docstrings" scope from the user rather than a
  minimal translate-only pass, since CLAUDE.md already mandates Google-style
  docstrings and these files were the last ones missing them.
- Left math in docstrings ASCII (`eps`, `p`, `O(eps^2)`) instead of Greek
  letters, to keep the "English-only / plain-ASCII in code" rule unambiguous.
- Kept the notebook's committed outputs (teaching value) and patched the two
  Vietnamese strings in place instead of stripping all outputs.
- Did **not** close TODO-0007 — that is the user's call (see the instruction
  note); flagged it as ready.

## Conclusion

Done. `grep -P '[\x{00C0}-\x{1EF9}]'` over `BasicML/**/*.{py,ipynb,md}` and
`BasicML/logs/` is now empty. Verification: `pyrefly check` shows no new errors
(3 pre-existing `Line2D.set_3d_properties` matplotlib-3d warnings unchanged);
`check_gradients.py` and `train_linear.py` run clean; all 6 translated demo
scripts run under an Agg + stubbed-`FuncAnimation` smoke harness;
`nbformat.validate` passes on the notebook.

Remaining under TODO-0007: only the standing "English commit messages"
convention, which is documented in CLAUDE.md and is not a tracked task.
