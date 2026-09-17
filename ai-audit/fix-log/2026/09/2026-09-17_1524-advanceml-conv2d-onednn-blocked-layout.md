---
id: FIX-0063
timestamp: 2026-09-17T15:24:00+07:00
todo_id: TODO-0043
---

## Prompt

User: "one Epoch is now took very long time ~300s or longer. Look into source code first, check for vectorize, kernel DNNL, SIMD implement. Optimze the core kernel first." — continuing from a prior session in this same conversation that had already opened TODO-0043 with a hypothesis (uncached `dnnl::primitive_desc` construction) but not yet implemented a fix.

## Action

- Stashed unrelated in-progress `feature/TODO-0042-resnet-depthwise-pointwise` work (with user's explicit go-ahead) and branched `fix/TODO-0043-onednn-primitive-cache` off `dev`.
- Implemented shape-keyed primitive caching (`gemm_cache()`, `pool_cache()`, `conv_cache()`, all `unordered_map`s keyed on actual tensor dims) in `AdvanceML/src/ops.cpp` for `matmul`, `pool2d` (fwd+bwd), and `conv2d` (fwd, `backward_data`, `backward_weights`) — this was TODO-0043's original item 1.
- Benchmarked the caching change in isolation (`pd_bench.cpp`, standalone, not part of the build): rebuild-per-call vs cached showed 0.75x–1.04x — no real speedup. The primitive-rebuild hypothesis was wrong.
- Used `DNNL_VERBOSE=1` to find the actual cause: `conv2d()`'s fixed `format_tag::nchw`/`oihw` memory descriptors force oneDNN into `jit_uni_ncsp_convolution:conv+ref:any`, an unoptimized reference-style fallback, instead of the AVX-512-VNNI blocked-layout kernel (`jit:avx512_core`). Confirmed `pool2d` and `matmul` were unaffected (they already select fast kernels with plain layouts).
- Fixed `conv2d`: `src_md`/`weights_md`/`dst_md` now use `format_tag::any` (forward, `backward_data`, `backward_weights`), letting oneDNN choose its optimal blocked layout; added `reorder_to`/`reorder_into` helpers doing explicit oneDNN reorders at the plain-NCHW/OIHW boundary, since `Tensor` only ever holds plain contiguous data.
- Isolated micro-benchmark on the actual conv2 layer shape (16→32ch, 28×28, batch 128): ~8ms/iter (`format_tag::any`) vs ~2100ms/iter (forced plain), ~180x.
- End-to-end confirmation on `train_cnn_mnist.cpp`: 445.59s/epoch before → 58.23s/epoch after (both measured this session under the same, noisy, multi-session system load), ~7.6x.
- `cmake --build AdvanceML/build` succeeds; `ctest --test-dir AdvanceML/build` passes 59/59, including the CNN forward/gradient-check tests, confirming the layout change didn't alter numerics.

## Decision

Kept the primitive caching from item 1 even though it wasn't the dominant cost — it's still correct, harmless, and guards against a real (if smaller) cost if shapes vary more in the future. Did not cache the new `reorder_to`/`reorder_into` primitives; their construction cost is small relative to the ~180x win from kernel selection, and adding another cache layer wasn't justified by evidence this session — left as a documented, cheap follow-up if profiling ever shows otherwise. Deferred TODO-0043's item 4 (scalar elementwise op vectorization audit) since the conv2d fix was clearly the dominant bottleneck; item 3 (confirm AVX-512 VNNI is used) is now answered as a byproduct (pooling/matmul already did, conv2d now does).

## Conclusion

Fixed/landed on `fix/TODO-0043-onednn-primitive-cache` (not yet merged to `dev`). Root cause was oneDNN kernel selection (forced plain memory format defeating AVX-512 VNNI blocked-layout kernels for convolution), not primitive-descriptor rebuild overhead as originally hypothesized in TODO-0043. ~7.6x measured end-to-end epoch-time improvement, ~180x on the isolated conv kernel. Follow-ups: a clean idle-machine re-measurement would tighten the end-to-end number; item 4 (elementwise op vectorization audit) still open in TODO-0043.
