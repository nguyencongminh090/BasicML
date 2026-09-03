# TODO Index (Archived)

Done or cancelled items, moved here from [../INDEX.md](../INDEX.md) to keep the active index short. IDs are never reused. See [../../README.md](../../README.md) for the schema.

| ID | Status | Priority | Source | Short Description | Instruction | Closed |
|----|--------|----------|--------|--------------------|-------------|--------|
| TODO-0001 | done | medium | user-report | MLP support: Sequential, init module, Linear bias, gradient check | — | 2026-09-02 |
| TODO-0002 | done | medium | user-report | L1/L2/ElasticNet regularization wired into optimizers | [TODO-0002](../../instructions/TODO-0002.md) | 2026-09-02 |
| TODO-0003 | done | medium | user-report | Example: regularization vs overfitting (train/val gap on make_moons) | [TODO-0003](../../instructions/TODO-0003.md) | 2026-09-02 |
| TODO-0004 | done | medium | user-report | Dropout as parameterless nn.Module + train()/eval() on Module/Sequential | [TODO-0004](../../instructions/TODO-0004.md) | 2026-09-02 |
| TODO-0005 | done | medium | user-report | Demo: animated MLP neuron/weight graph — heatmaps + sparse graph + gradient-flow / activation panels, plain vs Dropout+L2 | [TODO-0005](../../instructions/TODO-0005.md) | 2026-09-02 |
| TODO-0006 | done | medium | user-report | Demo: vanishing gradient — Sigmoid+Xavier vs ReLU+He deep MLP, per-layer gradient RMS + Π f'(z) animation | [TODO-0006](../../instructions/TODO-0006.md) | 2026-09-02 |
| TODO-0007 | done | medium | user-report | Internationalize repo to English + adopt code style (English-only, Clean-Code comments, Google docstrings, SOLID); CLAUDE.md, README, all examples/demo `.py` + notebook migrated | [TODO-0007](../../instructions/TODO-0007.md) | 2026-09-02 |
| TODO-0008 | done | medium | user-report | Add 12 activation layers (Identity/LeakyReLU/PReLU/ELU/SELU/Softplus/GELU/Swish/Mish/Hardtanh/Hardsigmoid/Softmax) with manual backward; core kept bare, CLAUDE.md style updated | [TODO-0008](../../instructions/TODO-0008.md) | 2026-09-03 |
| TODO-0009 | done | medium | user-report | More optimizers (Adam/AdamW/RMSProp/Adagrad/Adadelta/Nesterov) + `datasets.iter_minibatches` + comparison example; examples & 7/8 demos moved to Adam (vanishing-gradient demo kept on Momentum) | [TODO-0009](../../instructions/TODO-0009.md) | 2026-09-03 |
| TODO-0010 | done | medium | user-report | `AbsoluteLoss` (MAE/L1, correct subgradient) + `plot_dynamic_linear_abs.py` demo; `plot_dynamic_linear.py` back to plain fixed-lr loop; `plot_ewa.py` EWA demo | [TODO-0010](../../instructions/TODO-0010.md) | 2026-09-03 |
