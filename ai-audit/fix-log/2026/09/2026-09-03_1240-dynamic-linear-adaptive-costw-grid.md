---
id: FIX-0028
timestamp: 2026-09-03T12:40:00+07:00
todo_id: TODO-0010
---

## Prompt

"@BasicML/demo/plot_dynamic_linear.py Make adaptive space (grid) for Cost and
Weight" — panel 5 ("Cost vs Weight"). On the follow-up question the user chose
"Analytic curve on adaptive w-grid".

## Action

- Added `cost_vs_weight(x, y, b, w_range)` helper: sweeps a
  `np.linspace(*w_range, GRID_RESOLUTION)` of weight values with the bias held
  fixed and returns `(w_vals, cost_vals)` MSE along that grid. Google-style
  docstring, column-aligned assignments.
- `animate()` panel 5 (`ax_costw`): draws the analytic cost curve as a static
  gray background line using `b_opt` from `closed_form_optimum`; x-limits now
  `*w_range` (the same adaptive range the contour/3D panels use), y-limit
  `max(cost_curve.max(), history.cost.max()) * 1.1` instead of the fixed
  `history.weight.min()/max() ± 0.5` and `history.cost.max()`.
- The animated purple `(w, cost)` optimizer path is unchanged and rides on top.
- Not mine, already in the working tree at session start: `LEARN_RATE` 0.7→0.03,
  `Adam`→`SGD`, and the `Momentum`/`SGD` imports.

Verified: `pyrefly check` — no new errors on the file (the pre-existing
`Line2D.set_3d_properties` false-positive at :279 remains). Headless smoke run
(`MPLBACKEND=Agg`, `plt.show` stubbed) completes: final cost 0.0016.

## Decision

Fixed bias at `b_opt` (not the running bias) so the background curve is a
single stable parabola for the whole animation; reusing `w_range` keeps all
four `(w, ...)` panels on one consistent horizontal scale.

## Conclusion

Done. Panel 5 now shows the true cost-vs-weight parabola on an adaptive grid
with the optimizer path overlaid.
