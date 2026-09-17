---
id: FIX-0065
timestamp: 2026-09-17T18:01:16+07:00
todo_id: TODO-0045
---

## Prompt

"DO Task 0045": the AdvanceML op/kernel performance findings K1–K7 (conv2d layout conversions, scalar elementwise loops, fused softmax-CE, matmul transposes and bias, allocations, build flags, minor cleanups). The user signed off on a WIP-cap exception, chose to **implement** K1's "keep tensors blocked end to end" option, chose the aligned default-init `FloatBuffer` storage type (breaking API), and asked for one commit.

## Action

- Profiled a baseline first: an idle 10-epoch run under `/usr/bin/time -v`, `perf record -g` and an `ONEDNN_VERBOSE` breakdown (recorded in `ai-audit/instructions/TODO-0045.md`).
- New `AdvanceML/include/advanceml/buffer.hpp` (`AlignedDefaultInitAllocator`, `FloatBuffer`). New private `AdvanceML/src/detail/dnnl_layout.{hpp,cpp}`: engine/stream, interned layouts, cached reorders, in-place conversion to plain, tensor↔memory helpers.
- `tensor.hpp`/`tensor.cpp`: `Storage` gains `layout`; `data()`/`mutable_data()` return `FloatBuffer` and convert blocked storage to plain on read; new `has_plain_layout()`; `numel()` comes from the shape; accumulation handles layouts.
- `ops.cpp` rewrite:
  - conv2d keeps oneDNN layouts, reuses forward conversions in backward, converts `diff_dst` once, and keeps a per-entry `diff_weights` scratch buffer.
  - `batch_norm2d`/`1d`, `relu`, `global_avg_pool2d`/`global_max_pool2d` moved onto oneDNN primitives; pooling accepts blocked input.
  - matmul backward uses `permute_axes` views.
  - New ops `linear` (fused bias) and `softmax_cross_entropy`.
- Call sites: `Linear::forward` uses `linear`; new `SoftmaxCrossEntropyLoss`; `AdamW::step` loop hoisted onto raw pointers; `FloatBuffer` in optimizers, accuracy and mnist.
- Build: `-fno-math-errno` and the private `src` include directory in `AdvanceML/CMakeLists.txt`.
- `examples/train_cnn_mnist.cpp`: fused loss on logits, `GlobalAvgPool2D`, stale comment fixed.
- Tests: new `tests/kernel_perf_test.cpp` (9 tests); `autograd_engine_test.cpp`'s saved-input version test moved to `leaky_relu` plus a relu saved-output section; type fixes in 2 tests.
- `AdvanceML/README.md` performance and API notes updated.

## Decision

- **Layout lives on `Storage`, not `TensorImpl`**, because it describes the physical buffer that views share. Conversion to plain is lazy and in place, so every existing op, test and user reading `data()` keeps working without changes, and only the oneDNN-backed ops look at layouts. Interned layout pointers give cheap cache keys without hashing `memory::desc`.
- Backward closures don't keep non-owning `dnnl::memory` views over tensor storage, because a later plain read frees that buffer. They look it up again.
- BN and ReLU moved to oneDNN (rather than `#pragma omp` loops) because that is what lets the blocked layout flow through; it also multi-threads them.
- ReLU saves its output (`use_dst_for_bwd`), so a write to its input after `relu` no longer invalidates backward. This is correct, since backward doesn't read it, and the affected test was updated to an op that still saves its input.
- `FloatBuffer` ships with a constrained template `std::vector<float>` constructor, so user code and braced lists keep compiling.

## Conclusion

Fixed.
- **Tests:** `ctest` 77/77. The suite is also clean under ASan+UBSan with leak detection. Blocked vs forced-plain chain gives identical loss, running stats and gradients (1e-5), with both code paths confirmed via `ONEDNN_VERBOSE`. Transposed matmul still uses `gemm:jit:f32`.
- **Speed:** full idle run **75.13 s → 18.40 s (4.1x)**, deterministic run to run.
- **Memory:** max RSS with default malloc 166.7 → 209.0 MB. That is heap retention: with a fixed mmap threshold both binaries give 161.0 → 148.1 MB.
- **Trajectory:** train loss/accuracy trajectory matches closely (epoch 10: 0.0783/0.9820 → 0.0779/0.9821). Final test_acc 0.9644 → 0.9608, within the ±3% epoch-to-epoch test noise both runs show.
- **Follow-ups (not filed):** keep parameters in the blocked weight layout to remove the remaining weight reorders; use a pooling allocator if RSS retention matters.
