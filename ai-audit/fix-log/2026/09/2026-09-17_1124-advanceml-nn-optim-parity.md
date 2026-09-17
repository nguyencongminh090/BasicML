---
id: FIX-0062
timestamp: 2026-09-17T11:24:16+07:00
todo_id: TODO-0041
---

## Prompt

User said "DO TASK-0041" — bring AdvanceML's `nn`/`optim` C++ modules up to feature
parity with BasicML's broader Python equivalents: Loss (`AbsoluteLoss`,
`BinaryCrossEntropy`), Optimizer (`Nesterov`, `Adagrad`, `Adadelta`, `RMSprop`,
`Muon`), Layer (`GlobalAvgPool2D`, `GlobalMaxPool2D`, `BatchNorm1D`, `Dropout`),
Activation (`Tanh`, `Identity`, `PReLU`, `ELU`, `SELU`, `Softplus`, `Swish`,
`Mish`, `Hardtanh`, `Hardsigmoid`).

## Action

Starting exceeded the WIP cap (TODO-0011/TODO-0024 already in-progress) and the
scope was large (~14 classes) — asked the user via `AskUserQuestion`, who approved
the exception and chose one commit per category. Branched
`feature/TODO-0041-nn-optim-parity` off `dev` and made four commits:

1. **Loss** — `abs_loss`/`binary_cross_entropy` ops + `AbsoluteLoss`/`BinaryCrossEntropy` wrappers, `loss_parity_test.cpp` (2 tests).
2. **Optimizer** — `Nesterov`, `Adagrad`, `Adadelta`, `RMSprop` (all straightforward per-parameter state), and `Muon` (momentum + hand-rolled Newton-Schulz orthogonalization for 2D+ params, since optimizer state math doesn't need oneDNN), `optim_parity_test.cpp` (6 tests).
3. **Layer** — `global_avg_pool2d`/`global_max_pool2d` ops (hand-rolled reductions, not routed through oneDNN's fixed-kernel pooling primitive since the kernel spans the whole spatial plane and may be non-square), `batch_norm1d` op mirroring `batch_norm2d` over `(N, C)`, and an inverted-`dropout` op taking an owned `std::mt19937&` for its per-call mask; `GlobalAvgPool2D`/`GlobalMaxPool2D`/`BatchNorm1D`/`Dropout` wrappers, `layer_parity_test.cpp` (9 tests).
4. **Activation** — `Tanh`, `Identity`, `PReLU` (per-channel slope broadcasting over the trailing axis, matching `basicml.nn.activation.PReLU`), `ELU`, `SELU`, `Softplus`, `Swish`, `Mish`, `Hardtanh`, `Hardsigmoid` ops + wrappers, `activation_parity_test.cpp` (11 tests).

Each commit built clean (`cmake --build AdvanceML/build`) and kept `ctest` green
before proceeding to the next category; final state is 59/59 tests passing.
`pyrefly check` re-run at the end shows only the same pre-existing
`BasicML/examples/ref_code.py` errors as before this TODO — unaffected, as expected
for an AdvanceML-only change.

## Decision

Split into per-category commits (rather than one big commit or one commit per
class) per the user's explicit choice, balancing review granularity against the
number of commits for ~14 classes. Kept every new op inside `ops.hpp`/`ops.cpp`
rather than adding member `backward()` methods, per AdvanceML's autograd-graph
convention. Two ops (`global_avg_pool2d`/`global_max_pool2d`) don't call oneDNN,
matching the precedent set by the existing hand-rolled `batch_norm2d`/`flatten`
ops for cases the primitive library doesn't cleanly fit.

## Conclusion

All four categories implemented, tested, and committed on
`feature/TODO-0041-nn-optim-parity`. 59/59 Catch2 tests passing;
`cmake --build`/`ctest` both clean; `pyrefly check` confirmed unaffected. Branch
not yet merged into `dev` as of this entry.
