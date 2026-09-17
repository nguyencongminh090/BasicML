---
id: FIX-0066
timestamp: 2026-09-17T20:41:33+07:00
todo_id: TODO-0047
---

## Prompt

"DO TASK 0047" — the perf/memory findings (P0 profile first, P1 layout registry, P2 per-op overhead, P3 memory, P4 relu >6D fallback) from the same 2026-09-17 review of TODO-0044/0045 that also produced TODO-0046's correctness fixes.

## Action

Branched `feature/TODO-0047-onednn-profiling-perf` off `dev` (parked TODO-0024 to make WIP room, per user's choice). P0: `perf record -g` + `ONEDNN_VERBOSE=1` on `train_cnn_mnist`, plus an `OMP_NUM_THREADS` sweep, all recorded in `ai-audit/instructions/TODO-0047.md`. Then, in measured-cost order:

- P1: `AdvanceML/src/detail/dnnl_layout.cpp` — replaced the linear-scan layout registry with an `unordered_map<dnnl::memory::desc, shared_ptr<const Layout>, DescHash, DescEqual>` (hash approximates on dims/dtype/strides/inner-blocks, `==` is the correctness tiebreaker). New `AdvanceML/tests/dnnl_layout_test.cpp` (1,000-distinct-shape scaling test).
- P2: rejected — no code change (see Decision).
- P3 (malloc retention): `AdvanceML/include/advanceml/buffer.hpp` — added `detail::tune_malloc_once()` (`mallopt(M_MMAP_THRESHOLD, 8 * 1024 * 1024)`, glibc/Linux-guarded), called once from `AlignedDefaultInitAllocator::allocate()`. P3 (duplicate conv inputs): deferred, no code change.
- P4: `AdvanceML/src/ops.cpp` — `relu()` falls back to a new `relu_scalar()` plain loop for tensors beyond oneDNN's 6D descriptor limit. Test added to `AdvanceML/tests/kernel_perf_test.cpp` (7D forward + numerical-gradient backward).

Verification: `ctest --test-dir AdvanceML/build` 79/79. Before/after full 10-epoch `train_cnn_mnist` run (`OMP_NUM_THREADS=4`, same non-idle machine, back to back): wall time 16.05s -> 17.95s (within this session's run-to-run noise band), peak RSS ~205MB -> ~187MB (-9%), per-epoch loss/accuracy and final test accuracy (0.9652) bit-for-bit identical.

## Decision

P0's profile (`perf report --sort=dso`) showed the framework code P1/P2 target (`intern()`, per-op arg-map construction) at 0.18% self-time — far below this session's machine-noise floor — while oneDNN's own JIT kernels (57.8%) and `libgomp` OpenMP fork/join+spin-wait overhead (39.5%) dominate. P1 was implemented anyway (cheap, fixes a real unbounded-growth bug relevant to future variable-shape workloads, TODO already specced a test for it); P2 was rejected as not worth the risk for an unmeasurable win.

For P3, the TODO's own suggested `mallopt(M_MMAP_THRESHOLD, ...)` fix was implemented literally first at a low (128 KiB) threshold and found to cost 62% more wall time (16.05s -> 25.95s) for its ~148MB RSS win, because it forces this workload's ~6.1MB hot activation buffer through `mmap`/`munmap` on every call. Swept 128KiB/2/4/6/8 MiB and picked 8 MiB: above the hot buffer (so training speed is unaffected) while still fixing (as opposed to leaving dynamic) the threshold, which is what actually stops the RSS creep the TODO described. Documented as a doc comment on `tune_malloc_once()` so the tradeoff isn't silently "corrected" back to a smaller threshold later. Duplicate-conv-input dedup deferred: reworking what `conv2d` backward keeps alive for its saved-tensor check carries real autograd-correctness risk for a win the TODO's data never isolated from the malloc-retention number.

## Conclusion

Fixed / implemented: P1 (hashed layout registry), P3 malloc-retention (`mallopt`, tuned via measurement rather than the TODO's literal suggestion), P4 (relu >6D fallback). Rejected: P2 (profile shows no measurable win). Deferred: P3 duplicate-conv-input dedup (needs its own design pass). All P0-P4 items resolved with reasons recorded in `ai-audit/instructions/TODO-0047.md`. `ctest` 79/79 passing; training behavior unchanged (identical loss/accuracy trajectory and final test accuracy). Follow-up: the duplicate-conv-input memory item is a candidate for a future TODO if it's ever profiled as material.
