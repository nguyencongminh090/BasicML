---
id: FIX-0010
timestamp: 2026-09-02T20:25:00+07:00
todo_id: TODO-0005
---

## Prompt

"pyrefly installed" — user enabled the type checker; run it and clear anything
in the new demo files.

## Action

- Ran `pyrefly check`. Only finding in the session's new files:
  `plot_dynamic_mlp_graph.py:68-69` — `load_breast_cancer()` return typed as
  `tuple`, so `.data` / `.target` attribute access failed.
- Fixed: `X_raw, y_raw = load_breast_cancer(return_X_y=True)` then
  `np.asarray(...)` (identical to the fix already in
  `plot_dynamic_vanishing_gradient.py`).
- Re-ran: `plot_dynamic_mlp_graph.py` and `plot_dynamic_vanishing_gradient.py`
  → 0 pyrefly errors. Both demos re-run headless (50 epochs) OK.

## Decision

`return_X_y=True` gives a clean `(ndarray, ndarray)` tuple and avoids depending
on the sklearn `Bunch` type stub.

## Conclusion

Fixed. The 9 other repo-wide pyrefly errors (`examples/ref_code.py`,
`demo/plot_dynamic_3d_logistic.py`, `demo/plot_dynamic_linear.py`,
`demo/plot_dynamic_logistic.py`, `demo/layer_space_transformation.ipynb`) are
pre-existing and out of scope for this session.
