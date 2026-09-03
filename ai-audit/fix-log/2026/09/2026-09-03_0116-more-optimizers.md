---
id: FIX-0018
timestamp: 2026-09-03T01:16:24+07:00
todo_id: TODO-0009
---

## Prompt

"Write more Optimizer — MiniBatch, Adam, RMSProp, ... and more". Follow-up
AskUserQuestion answers: full optimizer scope (Adam + RMSProp + Adagrad +
Adadelta + AdamW + Nesterov), a "Mini-batch Gradient Descent" note for the
MiniBatch item, and yes to a comparison example.

## Action

Branch `feature/TODO-0009-more-optimizers` off `dev`.

- Added `BasicML/basicml/optim/{adagrad,rmsprop,adadelta,adam,adamw,nesterov}.py`
  — one `Optimizer` subclass each, `Momentum`-style (per-parameter state list,
  regularizer grad folded in, `requires_grad and grad is not None` guard,
  `zero_grad()` loop). Bare core style: no comments, no docstrings.
- Added `BasicML/basicml/datasets/batching.py` with `iter_minibatches(X, y,
  batch_size, shuffle=True, drop_last=False, random_state=None)` generator
  (Google-style docstring — documented package edge); exported it from
  `datasets/__init__.py`.
- Added `BasicML/examples/train_optimizer_comparison.py` — same MLP / init /
  data, mini-batch GD (`iter_minibatches`, one step per batch), one log-scale
  training-BCE curve per optimizer on `make_moons`.

## Decision

MiniBatch is a data-batching concern, not an `Optimizer` — implemented it as a
generator in the `datasets/` edge package rather than a fake optimizer class, so
`optim/` stays purely optimizers and the mini-batch loop lives in the training
script where it belongs. Each optimizer is its own file / subclass (open-closed,
matches `sgd.py` / `momentum.py`), not flags on a mega-class. AdamW takes an
explicit `weight_decay` (decoupled) rather than relying on an L2 `Regularizer`.
Nesterov uses the Sutskever look-ahead form so it reuses the same velocity-list
state shape as `Momentum`.

## Conclusion

Fixed. `MPLBACKEND=Agg python3.13 BasicML/examples/train_optimizer_comparison.py`
— all 8 optimizers converge (final BCE 0.042–0.054, all reach BCE < 0.30 within
2 epochs). `pyrefly check` — no new errors in the added files (9 pre-existing
errors in `examples/ref_code.py` and `demo/plot_dynamic_*` untouched).
Follow-up: commit via `/make-commit`, merge to `dev`, close TODO-0009.
