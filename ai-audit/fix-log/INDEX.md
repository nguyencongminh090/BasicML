# Fix-Log Index

Chronological, append-only — newest entry at the bottom. Read this table first; open the linked file only for the full Prompt/Action/Decision/Conclusion detail. See [../README.md](../README.md) for the schema.

| ID | Timestamp | Summary | TODO | File |
|----|-----------|---------|------|------|
| FIX-0001 | 2026-09-02T16:35:26+07:00 | Add `code_author` / `ai_role` authorship fields to TODO schema (README + TEMPLATE), set on TODO-0002 | TODO-0002 | [2026/09/2026-09-02_1635-todo-authorship-fields.md](2026/09/2026-09-02_1635-todo-authorship-fields.md) |
| FIX-0002 | 2026-09-02T16:54:34+07:00 | Add `examples/train_regularization.py` — over-capacity MLP on noisy make_moons, no-reg vs L2 vs L1 train/val gap | TODO-0003 | [2026/09/2026-09-02_1654-regularization-example.md](2026/09/2026-09-02_1654-regularization-example.md) |
| FIX-0003 | 2026-09-02T17:22:02+07:00 | Improve `train_regularization.py` — revert hidden layers to ReLU (fix L2 collapse), add learning-curve history + summary table | TODO-0003 | [2026/09/2026-09-02_1722-regularization-example-improve.md](2026/09/2026-09-02_1722-regularization-example-improve.md) |
| FIX-0004 | 2026-09-02T17:29:38+07:00 | `train_regularization.py` — translate to English, add val-set boundary plot row with misclassified markers | TODO-0003 | [2026/09/2026-09-02_1729-regularization-example-english-valplot.md](2026/09/2026-09-02_1729-regularization-example-english-valplot.md) |
| FIX-0005 | 2026-09-02T18:50:40+07:00 | `Module` gets `__init__`/`training` + `train()`/`eval()`; `super().__init__()` added to Linear, Sigmoid/ReLU/Tanh, Sequential, Dropout | TODO-0004 | [2026/09/2026-09-02_1850-module-training-state.md](2026/09/2026-09-02_1850-module-training-state.md) |
| FIX-0008 | 2026-09-02T19:55:00+07:00 | Add `demo/plot_dynamic_vanishing_gradient.py` — Sigmoid+Xavier vs ReLU+He deep MLP, animated per-layer gradient RMS + `Π f'(z)` showing vanishing | TODO-0006 | [2026/09/2026-09-02_1955-vanishing-gradient-demo.md](2026/09/2026-09-02_1955-vanishing-gradient-demo.md) |
| FIX-0009 | 2026-09-02T20:15:00+07:00 | `plot_dynamic_vanishing_gradient.py` — add `backward_capture` + `RMS(δ=∂L/∂z)` per-depth panel; now shows the textbook vanishing staircase | TODO-0006 | [2026/09/2026-09-02_2015-vanishing-delta-measure.md](2026/09/2026-09-02_2015-vanishing-delta-measure.md) |
