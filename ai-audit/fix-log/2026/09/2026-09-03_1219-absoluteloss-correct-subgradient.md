---
id: FIX-0026
timestamp: 2026-09-03T12:19:02+07:00
todo_id: TODO-0010
---

## Prompt

"Update @BasicML/basicml/nn/loss.py cho AbsoluteLoss, sử dụng đúng chuẩn
gradient."

(Follow-up to FIX-0023, which had deliberately given AbsoluteLoss the wrong
MSE-style gradient. User now wants the correct one.)

## Action

- `basicml/nn/loss.py`: `AbsoluteLoss.backward` now returns the proper MAE
  subgradient `np.sign(y_pred - y_true) / n` instead of `(y_pred - y_true) / n`.
  Forward (`mean|y_pred - y_true|`) unchanged.
- `demo/plot_dynamic_linear_abs.py` — brought the prose in line with the now
  correct loss:
  - module docstring: no longer "deliberately WRONG"; explains that the
    constant-magnitude subgradient means a fixed lr jitters near the optimum
    instead of settling, vs MSE which lands smoothly.
  - `train_and_record` docstring: "(wrong) MSE-style" -> "proper subgradient
    sign(y_pred - y) / n".
  - `LEARN_RATE` comment rewritten (bounded subgradient, no overflow, but fixed
    lr => jitter).
  - window title "(WRONG Absolute Loss)" -> "(Absolute / MAE Loss)".

## Decision

The demo's whole point is now to *show* the L1 pathology rather than mask it, so
the docstrings had to flip too. Kept `LEARN_RATE = 0.01` / `EPOCHS = 400`: with
`sign` gradient this reaches the optimum region then oscillates visibly. Left the
unused `Adam` import for toggling.

## Conclusion

Done. Only caller of `AbsoluteLoss` is that demo (blast radius checked).
`pyrefly check`: only the pre-existing `Line2D.set_3d_properties` false
positives. Headless run: MAE optimum ~0.011, but last-10 epoch costs bounce
0.085 <-> 0.23 — the constant-gradient non-convergence is now on screen.
