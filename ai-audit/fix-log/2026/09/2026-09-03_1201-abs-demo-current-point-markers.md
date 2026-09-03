---
id: FIX-0024
timestamp: 2026-09-03T12:01:02+07:00
todo_id: TODO-0010
---

## Prompt

"@plot_dynamic_linear_abs.py add red point for J(w,b), w, b for me to track the
last point."

## Action

`BasicML/demo/plot_dynamic_linear_abs.py` — added a red current-position marker
to every trajectory panel, updated each frame in `update()`:

- `curve_dot` on the learning curve at `(frame, J)`
- `costw_dot` on the cost-vs-weight panel at `(w, J)`
- `path_dot` on the 2D cost-surface path at `(w, b)`
- `path_dot_3d` on the 3D surface path at `(w, b, J)`

`ax_curve` title now also reads `J(w, b) = ... | w = ..., b = ...`. All four
new artists returned from `update()`.

## Decision

Reused the existing `history` arrays and the `blit=False` FuncAnimation; a
per-frame `set_data` on four extra Line2D/Line3D artists is the lightest way to
show "where am I now" without touching the training code.

## Conclusion

Added. `pyrefly check`: only the pre-existing `Line2D has no set_3d_properties`
false-positive (now also on the new `path_dot_3d`, same cause as the existing
`path_line_3d` warning). Headless `animate()` builds without error.
