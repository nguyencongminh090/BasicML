---
id: FIX-0058
timestamp: 2026-09-17T05:32:03+07:00
todo_id: TODO-0038
---

## Prompt

User: "DO TASK 0034" — TODO-0034 is an umbrella tracking TODO for AdvanceML
component-library parity with BasicML, not directly actionable itself. Asked which
of the three remaining areas (CNN layers / datasets-metrics / MNIST example) to
start, and to approve a WIP-cap exception (TODO-0011 and TODO-0024 both already
in-progress, cap is 2). User chose CNN layers and approved the exception.

## Action

Filed TODO-0038 (sub-TODO of TODO-0034, area 1), branched
`feature/TODO-0038-cnn-layers` off `dev`. Read the existing AdvanceML architecture
(`tensor.hpp`, `module.hpp`, `linear.hpp/cpp`, `ops.hpp/cpp`) to match its
established pattern (free-function autograd ops in `ops.hpp`/`ops.cpp`, thin
`nn::Module` wrapper classes, oneDNN engine/stream as static helpers in
`ops.cpp`'s anonymous namespace — no separate `Backend` interface exists yet).

Implemented and added to `AdvanceML/include/advanceml/ops.hpp` /
`AdvanceML/src/ops.cpp`:
- `conv2d` via `dnnl::convolution_forward` / `convolution_backward_data` /
  `convolution_backward_weights`.
- `max_pool2d` / `avg_pool2d` via `dnnl::pooling_forward`/`pooling_backward`
  (`pooling_max` with workspace; `pooling_avg_exclude_padding` without).
- `flatten` — pure reshape, no oneDNN.
- `batch_norm2d` — hand-written per-channel normalization (no oneDNN primitive),
  `running_mean`/`running_var` passed by mutable reference as plain buffers.

Added `nn::Conv2D`, `nn::MaxPool2D`, `nn::AvgPool2D`, `nn::Flatten`,
`nn::BatchNorm2D` module wrappers (new header+source pairs under
`AdvanceML/include/advanceml/nn/` and `AdvanceML/src/nn/`), wired into
`AdvanceML/CMakeLists.txt`. Added `AdvanceML/tests/cnn_test.cpp` (9 Catch2 cases:
forward hand-computed values + numerical-gradient checks per op, BatchNorm2D
train/eval-mode checks, an end-to-end Conv2D/ReLU/MaxPool2D/Flatten/Linear
`Sequential` training test), registered in `AdvanceML/tests/CMakeLists.txt`.

Committed as `4e59fe6` on `feature/TODO-0038-cnn-layers`.

## Decision

Followed the codebase's existing ops/Module split rather than inventing a new
`Backend` abstraction, since none exists yet in the actual code (the umbrella
TODO's wording assumed one). Implemented `BatchNorm2D` without a oneDNN primitive
since the umbrella only mandated oneDNN for conv/pooling; a manual per-channel
implementation is simpler and easier to verify via numerical gradient checks.
Chose symmetric stride/padding (single scalar, not per-axis pairs) for `conv2d`
and pooling ops to keep the API surface small, matching this project's
from-scratch teaching-adjacent scope.

## Conclusion

Fixed. `cmake --build AdvanceML/build` compiles clean; `ctest --test-dir
AdvanceML/build` — all 24 tests pass (15 pre-existing + 9 new). TODO-0038 complete.
TODO-0034 umbrella still open: areas 5 (datasets/metrics) and 6 (MNIST CNN
example) remain.
