---
id: FIX-0007
timestamp: 2026-09-02T19:35:00+07:00
todo_id: TODO-0005
---

## Prompt

"Thông tin nào đại diện cho giá trị weight trong model? ... a large model may have
Vanishing when forward/backprop and want to see how weight work by visualize."
User chose (multi-select) to extend the existing `plot_dynamic_mlp_graph.py`
with: per-layer gradient-norm panel, heatmap toggle to gradient view, and
per-layer ReLU activation stats.

## Action

`BasicML/demo/plot_dynamic_mlp_graph.py`:
- Added `weight_grad(layer)` helper (narrows `Tensor.grad: Optional[np.ndarray]`).
- `train_and_record` now records every `RECORD_EVERY` epochs, in addition to
  weights: `grad` (list of `∂L/∂w` matrices), `gnorm` (Frobenius norm per Linear
  layer, read before `optimizer.zero_grad()`), `act_mean` (mean|ReLU out|) and
  `act_dead` (% zero units) from an eval-mode forward on the train set.
- `animate`: gridspec 3×2 → 4×2 (FIG_SIZE 18×10 → 18×13). New `ax_grad`
  (log-y `‖∂L/∂w‖`, one line per layer, plain solid / reg dashed, viridis by
  depth) and `ax_act` (grouped bars mean|act| log-y + %dead text labels).
- Heatmap rows now switch between weights and `|∂L/∂w|` via `HEATMAP_MODE`
  constant and a `key_press_event` handler (`g` / `w`); suptitle shows the mode.
- Docstring updated to describe panels 2–4 and note the gradient shown is the
  pure backprop gradient (L2 term is applied by the optimizer, not in `w.grad`).

## Decision

Kept everything in one file per the user's choice — the weight-graph and the
gradient-flow views share the same two trained models and history, so splitting
would duplicate the training. Gradient norm over epochs on a log axis is the
standard way to reveal vanishing (early-layer lines sitting orders of magnitude
below later ones); activation stats give the forward-side cause (dead ReLU /
saturation). A key toggle was preferred over a matplotlib Button widget to keep
the dependency surface and layout unchanged.

## Conclusion

Fixed. Smoke-tested headless (matplotlib Agg, `FuncAnimation` patched to force
`update()` on first/middle/last frames, sample frame rendered to PNG and checked)
in both `HEATMAP_MODE="weight"` and `"grad"`: all four rows populate, curves,
grad-norm lines, activation bars and the neuron graph update correctly.
`pyrefly` not installed here — static check still pending on the user's machine.
On `feature/TODO-0005-mlp-graph-demo`, not yet committed.
