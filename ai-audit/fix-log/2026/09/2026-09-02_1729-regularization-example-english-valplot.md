---
id: FIX-0004
timestamp: 2026-09-02T17:29:38+07:00
todo_id: TODO-0003
---

## Prompt

"1. Language in code/comment should use English. 2. Add one val plot, show how model
behave on val sets." (re `BasicML/examples/train_regularization.py`)

## Action

- Translated the module docstring and all comments/strings in `train_regularization.py`
  from Vietnamese to English.
- Plot upgraded from 2xN to 3xN: row 1 = decision boundary over the training set,
  row 2 = the same trained model evaluated over the held-out validation set with `x`
  markers on misclassified val points (`plot_boundary(..., mark_wrong=True)`),
  row 3 = train vs val learning curves. Extracted `_boundary_grid` helper.

## Decision

- Repo default is Vietnamese documentation (CLAUDE.md), but the user explicitly asked for
  English on this example -> user override, applied to this file only. Other examples left
  as-is.
- Val behaviour shown as a boundary-over-val-points panel (not a separate metric) so the
  jagged vs smooth contrast from row 1 lines up visually with where val errors land.

## Conclusion

Done. Ran under `MPLBACKEND=Agg`, exit 0. Numbers unchanged from FIX-0003
(no-reg 1.000/0.794, L2 0.875/0.845, L1 0.900/0.843). `pyrefly check` not run (not on
PATH) -- user to run before `/make-commit`.
