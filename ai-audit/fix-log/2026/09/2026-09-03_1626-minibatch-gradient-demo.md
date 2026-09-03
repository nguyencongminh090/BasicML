---
id: FIX-0029
timestamp: 2026-09-03T16:26:01+07:00
todo_id: TODO-0011
---

## Prompt

"Create demo plot of mini-batch instead of SGD. I would like to see gradient at
each batch vs full batch (BatchGD). I have just been presented about Momentum,
that I need gradient to be stable. I want to see how gradient change direction
between each batch."

## Action

- Added `BasicML/demo/plot_dynamic_minibatch_gradient.py` (new, ~300 lines):
  - Synthetic linear dataset `y = 2.5x + 4 + N(0, 2.5)`, N=256, batch_size=32
    (8 mini-batches); `data.csv` too small at 9 rows.
  - `mse_gradient(w, b, x, y)` helper matching `MSELoss` convention
    (`dJ/dw = mean(x·resid)`, `dJ/db = mean(resid)`).
  - `train_and_record`: mini-batch SGD and Momentum run in lock-step on the same
    per-epoch `iter_minibatches` schedule; Momentum lr scaled by `(1 - momentum)`.
    Each epoch freezes the SGD point and records the full-batch descent
    direction, the per-batch descent directions, and the per-batch angle to the
    full-batch direction.
  - `animate`: 4 panels — (1) MSE cost contour with the gradient fan (bold
    crimson full-batch arrow + thin blue per-batch fan) and both optimizer
    paths; (2) current fit vs ground truth; (3) full-batch learning curves;
    (4) per-batch angle bar chart with live mean/max in the title.
- Updated the demo table in `README.md` with a row for the new script.
- Recorded session notes in `ai-audit/instructions/TODO-0011.md`; created
  `ai-audit/todo/TODO-0011.md` + INDEX row.

## Decision

- Gradient fan chosen (over side-by-side paths only) per user's answer: it makes
  the batch-to-batch direction scatter literally visible at one point.
- Gradients for the fan computed by a local closed-form helper rather than
  spinning up a scratch `Linear` per batch — transparent for a teaching demo and
  avoids `zero_grad` bookkeeping.
- Briefly added a green "Momentum average-step" arrow, then removed it: near
  convergence the accumulated velocity is dominated by stale noise and points
  away from the optimum, which would mislead rather than teach.

## Conclusion

Fixed. `pyrefly check` reports no errors in the new file (the pre-existing 11
errors in `examples/ref_code.py` etc. are untouched). Rendered headless across
epochs: fan spread ~20° early (all batches agree, arrows toward the min) vs
~150° near convergence (batches disagree once the full-batch gradient shrinks) —
the intended contrast. Both optimizer paths converge to full-batch MSE ≈ 2.81
(≈ noise variance / 2). Committed on
`feature/TODO-0011-minibatch-gradient-demo`.
