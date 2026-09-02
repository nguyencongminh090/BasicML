---
id: FIX-0008
timestamp: 2026-09-02T19:55:00+07:00
todo_id: TODO-0006
---

## Prompt

"Hãy tạo một file tương tự (you can copy) nhưng đổi setup cho tôi thấy vanishing"
— after establishing that the TODO-0005 demo (ReLU + He, 4 hidden layers) shows
no vanishing.

## Action

Added `BasicML/demo/plot_dynamic_vanishing_gradient.py` (~380 lines), adapted from
`plot_dynamic_mlp_graph.py`:
- `MODELS` = two configs, same `30 → 12×24 → 1` shape: `sigmoid + xavier` vs
  `relu + he`; Sigmoid output on both; no dropout / no regularization.
- `build_model(n_features, act, init_type)`, `hidden_activations()` (activation
  layers minus the output Sigmoid), `act_out()` (safe `.out` accessor),
  `act_derivative_mean()` (reconstructs `f'(z)` from cached `.out`).
- `train_and_record()` — full-batch manual loop; every `RECORD_EVERY` epochs
  snapshots weights, `∂L/∂w` matrices, per-layer `RMS(∂L/∂w)` (read before
  `zero_grad()`), per-hidden-layer `mean|activation|` and `mean f'(z)`.
- `animate()` — 4×2 gridspec: two neuron graphs; a row of 6 evenly-sampled
  Linear-layer heatmaps per model (weight ↔ gradient toggle, `g`/`w` key,
  default `grad`), each subplot titled with its norm; BCE loss; val accuracy;
  `RMS(∂L/∂w)` vs depth (log y, current epoch, both models); `mean f'(z)` vs
  depth with cross-layer product `Π f'(z)` in the legend and a 0.25 guide line.
- Suptitle shows `‖g‖_L1 / ‖g‖_last` for each model.
- No changes to `basicml/`.

## Decision

Kept the same visual language as TODO-0005 so the two demos read as a pair.
Chose Sigmoid + Xavier over Tanh as the vanishing case (deriv ≤ 0.25, not
zero-centred — worst case, clearest teaching signal). Used RMS rather than
Frobenius norm for the depth profile so the 30×24 / 24×24 / 24×1 layers are
comparable on one axis. Dropped dropout/L2 entirely — this demo is about
gradient flow, and dropout noise would muddy the per-layer gradient reading.
`f'(z)` is reconstructed from cached activations rather than adding a method to
the `Activation` classes (no `basicml/` change).

## Conclusion

Fixed. Smoke-tested headless (matplotlib Agg, `FuncAnimation` patched to force
`update()` on first/middle/last frames, frame PNG rendered and checked) in both
`HEATMAP_MODE`. 400-epoch run: sigmoid model loss ~0.65 flat / val acc 60.2%
(majority class); relu model loss→0 / val acc 98.2%. Gradient RMS ~1000× lower
for sigmoid across every layer; `Π f'(z)` ≈ 2.5e-8 (sigmoid) vs 6.3e-5 (relu) —
vanishing is clearly visible. `pyrefly` not installed in this environment.
Uncommitted together with TODO-0005 on `feature/TODO-0005-mlp-graph-demo`.
