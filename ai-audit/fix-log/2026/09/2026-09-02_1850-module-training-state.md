---
id: FIX-0005
timestamp: 2026-09-02T18:50:40+07:00
todo_id: TODO-0004
---

## Prompt

User (after implementing `Module.__init__` themselves) chose the `__init__` +
`super().__init__()` approach for the new `training` flag over a bare class attribute,
and asked: "Could you help me add and update all something related? You can use codegraph."

## Action

Used codegraph to enumerate every `Module` subclass, then:

- `basicml/nn/module.py` — kept `__init__(self)` setting `self.training = True`; fixed
  `train(self, mode: bool = True)` (added default) and `eval()` (now `return self.train(False)`).
- `basicml/nn/linear.py` — `Linear.__init__`: added `super().__init__()` as first line.
- `basicml/nn/activation.py` — `Sigmoid`, `ReLU`, `Tanh` `__init__`: added `super().__init__()`.
  (`Activation` is a bare `pass` marker subclass — inherits `Module.__init__`, no change.)
- `basicml/nn/sequential.py` — `Sequential.__init__`: added `super().__init__()`; also its
  `train()` override (set own flag + fan out to `self.layers`) and removed a stray blank line.
- `basicml/nn/dropout.py` — `Dropout.__init__`: added `super().__init__()`.

`nn/__init__.py` left empty (repo convention: examples import layers by full module path).

## Decision

`__init__` + `super().__init__()` over a class attribute: user's explicit choice, matches
PyTorch's `nn.Module` contract and keeps `training` a normal instance attribute visible to
type checkers. The cost — every subclass constructor must call `super().__init__()` — was
paid across all 6 subclasses in this change so no layer reads an unset `self.training`.

## Conclusion

Applied. Verified with `python3.13` (numpy only available under that interpreter here):

- `examples/check_gradients.py` — all 4 cases OK (max rel err ~1e-7).
- `examples/train_linear.py`, `train_logistic.py`, `train_regularization.py` — all run,
  outputs unchanged.
- Smoke test: `Sequential(Linear, ReLU, Dropout(0.5), Linear, Sigmoid)` — `train()`/`eval()`
  fan out to all layers; Dropout stochastic across forwards in train, deterministic
  passthrough with `mask is None` in eval; `backward` runs in both modes.

`pyrefly check` NOT run (pyrefly not installed in this environment) — user to run before merge.
Follow-ups: Dropout not added to `check_gradients.py` (stochastic mask breaks numeric check);
separate TODO for a Dropout train/val-gap example.
