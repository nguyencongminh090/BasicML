---
id: FIX-0009
timestamp: 2026-09-02T20:15:00+07:00
todo_id: TODO-0006
---

## Prompt

"okay you can add measure/activation" — after the user noted the
`RMS(∂L/∂w)` vs depth panel in `plot_dynamic_vanishing_gradient.py` was flat and
did not show the expected depth staircase.

## Action

`BasicML/demo/plot_dynamic_vanishing_gradient.py`:
- Added `backward_capture(model, grad_out)` — same loop as `Sequential.backward`
  (weights still accumulate into `w.grad`) but returns `delta = ∂L/∂z` at the
  input of each Linear layer, ordered input→output. Training loop now calls it
  instead of `model.backward(...)`.
- New history key `dnorm` = per-layer `RMS(delta)`, recorded every
  `RECORD_EVERY` epochs.
- Bottom-left panel: plots `RMS(delta)` (solid + markers) per model as the
  primary curve, with `RMS(∂L/∂w)` kept as a faint dotted line. Title →
  "Tin hieu backprop theo do sau (delta = ∂L/∂z)". y-limits now span both
  quantities.
- Suptitle ratio changed from `‖g‖_L1/‖g‖_last` to `‖delta_L1‖/‖delta_last‖`.
- Docstring panel-4 description updated.
- No `basicml/` change.

## Decision

`∂L/∂w_l = a_{l-1}^T @ delta_l`, so the forward activations `a` partly mask the
depth decay in `∂L/∂w`; `delta = ∂L/∂z` is the pure backprop signal and is the
correct quantity for visualizing vanishing. Captured by re-implementing the
`Sequential.backward` loop in the demo rather than adding a hook to the library.
Kept the `∂L/∂w` line (dotted) so the contrast — why the earlier panel looked
flat — is visible on the same axes.

## Conclusion

Fixed. 400-epoch smoke run: sigmoid `RMS(delta)` is now a clean straight
log-line from ~3e-12 at layer 1 to ~1e-3 at layer 13 (~10 orders across depth);
relu `RMS(delta)` flat ~1e-6. Ratio `‖delta_L1‖/‖delta_last‖`: sigmoid 2.5e-9 vs
relu 0.77. Smoke-tested headless in both `HEATMAP_MODE`. `pyrefly` not installed
here. Uncommitted on `feature/TODO-0005-mlp-graph-demo` with the rest of
TODO-0005 / TODO-0006.
