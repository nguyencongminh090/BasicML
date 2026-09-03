---
id: FIX-0023
timestamp: 2026-09-03T11:58:19+07:00
todo_id: TODO-0010
---

## Prompt

"@BasicML/basicml/nn/loss.py I would like to add a wrong loss function:
absolute function |x| and add a wrong plot_dynamic_linear_abs.py"

Clarified: apply the same training setup as `plot_dynamic_linear.py`; the demo
copies that file and swaps loss + cost surface; user will watch the behaviour
themselves.

## Action

- `basicml/nn/loss.py`: added `AbsoluteLoss(Loss)`. `__call__` returns the true
  mean absolute error `mean|y_pred - y_true|`; `backward()` deliberately returns
  the MSE-style gradient `(y_pred - y_true) / n` instead of the correct
  subgradient `sign(y_pred - y_true) / n` — the "wrong" part (differentiating
  `|x|` as if it were `x^2/2`). No comments/docstrings, per core-lib convention.
- New `BasicML/demo/plot_dynamic_linear_abs.py`: copy of `plot_dynamic_linear.py`
  with `MSELoss` -> `AbsoluteLoss`; `closed_form_optimum` replaced by
  `l1_optimum` (dense local grid search, seeded at the least-squares solution);
  `cost_surface` now `mean|...|`; panel labels/titles relabelled MSE -> MAE;
  window title marked "(WRONG Absolute Loss)"; module docstring explains the
  forward/backward mismatch.

## Decision

User explicitly wants a teaching artefact showing a mis-derived gradient, and
said they will observe the run themselves, so the wrongness is left in the
`backward()` (not the forward) and the training hyper-params match
`plot_dynamic_linear.py` unchanged. `l1_optimum` via grid search because the MAE
surface has no simple closed form.

## Conclusion

Added. `pyrefly check` clean for both files (pre-existing Line2D
`set_3d_properties` warning in the demo is unchanged from the original).
Headless run: training completes; on this near-noiseless dataset the wrong
MSE-style gradient still lands near the MAE optimum (`w~2.00, b~0`), final
cost ~0. Visual dynamics are for the user to inspect.
