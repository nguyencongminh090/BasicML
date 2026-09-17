---
id: FIX-0064
timestamp: 2026-09-17T17:12:00+07:00
todo_id: TODO-0044
---

## Prompt

User: "DO Task 0044", the AdvanceML autograd-engine performance review findings A1–A6. User decisions: close TODO-0043 first and branch off `dev`; WIP-cap exception; A6 = shared whole-buffer `Storage` (no strides); A5 = new `Tensor::mutable_data()` with `data()` const-only; all items in one commit.

## Action

- `include/advanceml/tensor.hpp` / `src/tensor.cpp`:
  - Added `Storage` (buffer + version), `SavedVersion`, and an abstract `Node` with fixed-size `inputs`/`needs_input_grad`/`saved` arrays, `apply()`, `release()`, `check_saved_versions()`, and an iterative destructor.
  - Added `TensorImpl::visit_mark`/`pending_grad`, `NoGradGuard`/`is_grad_enabled()`, `Tensor::mutable_data()`/`version()`, and `backward(bool retain_graph = false)`.
  - Rewrote `backward()`: iterative topological sort that prunes edges needing no gradient, per-tensor pending-gradient slots instead of hash maps, gradient buffers adopted and summed in place only when uniquely owned, and each node released as it runs.
- New `include/advanceml/detail/autograd.hpp`: `LambdaNode<F>` (backward lambda stored inside the node), `should_record()`, `record()`, `view()`.
- `src/ops.cpp`: every op moved to `should_record`/`record` and skips unneeded input gradients (conv2d `backward_data`, loss `grad_target`, and so on). `conv2d`/`pool2d` use `forward_inference` when not recording, with the prop kind added to the cache key. BN and swish skip backward-only buffers. `flatten`/`identity`/eval `dropout` are storage-sharing views. `sigmoid`/`tanh`/`softmax` save the output storage once.
- `src/optim/*.cpp`: one `mutable_data()` reference per parameter per step. Tests' numerical-gradient helpers write through `mutable_data()`. `examples/train_cnn_mnist.cpp` `evaluate()` uses `NoGradGuard`. `ops.hpp` docs and the README gained an Autograd section.
- New `tests/autograd_engine_test.cpp` (9 test cases).

## Decision

- Stored the backward lambda inside a templated node, rather than hand-writing one node class per op: one allocation instead of two, and ops stay readable.
- `zero_grad()` keeps resetting the gradient and relies on adopting the buffer, so `has_grad()` semantics don't change.
- In-place gradient accumulation is guarded by `use_count` checks, because gradients and views legitimately share buffers.
- `retain_graph = false` drops edges and saved state per node, so activation memory is freed during backward. This matches PyTorch's second-backward error.
- Deferred A4's inline shape and single-allocation tensor: `shape()` is public API, and those allocations weren't measured as a bottleneck.

## Conclusion

Done.
- 68/68 ctest, plus clean ASan/UBSan/LSan runs of the whole suite.
- Full 10-epoch `train_cnn_mnist` under non-idle load (5.7–9.2): about 2x faster per epoch (baseline ~14–18 s on quieter epochs vs 7–8 s) and max RSS 242.7 MB → 163.4 MB (−33%).
- In the default build, the trajectory differs in the 4th decimal from epoch 2 onward. That was bisected to GCC FMA contraction (`-ffp-contract=fast`) changing with loop codegen. With `-ffp-contract=off` on both builds, train loss/accuracy are identical to 8 decimals, so gradients are numerically unchanged.
- Follow-ups: idle-machine re-measurement (TODO-0045 profiling baseline), deferred A4 items.
