# AdvanceML

A from-scratch, performance-oriented deep learning library in modern C++ (C++20/23), with a real autograd engine — a define-by-run computation graph and `.backward()`, unlike [`BasicML`](../BasicML)'s hand-written per-layer `backward()`. GEMM/conv-heavy ops are backed by Intel oneDNN, which targets this machine's AVX-512 VNNI.

This is a sibling project to `BasicML`, not a replacement: `BasicML` stays a from-scratch, manual-backward, pure-NumPy teaching project. `AdvanceML` exists because that design cannot run CNN/Transformer-scale workloads efficiently on CPU.

See root `CLAUDE.md`'s `## AdvanceML` section for full architecture/style rules, and `ai-audit/instructions/TODO-0030.md` for the reasoning behind the language/backend/build-system choices, including why hand-written kernels aren't expected to beat oneDNN/PyTorch on raw GEMM throughput.

## Build

Requires a C++20/23 compiler, CMake, Ninja, and oneDNN development headers (`libdnnl-dev` on Debian/Ubuntu).

```bash
cmake -B build -S . -G Ninja
cmake --build build
ctest --test-dir build
```

Catch2 (test framework) is fetched automatically via CMake `FetchContent` — nothing to install manually for tests.

## Examples

`train_cnn_mnist` — a `(Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D) x2 -> Conv2D -> BatchNorm2D -> ReLU -> GlobalAvgPool2D -> Flatten -> Linear` CNN trained on real MNIST with `AdamW` and the fused `SoftmaxCrossEntropyLoss`, the AdvanceML counterpart to [`BasicML/examples/train_cnn_mnist.py`](../BasicML/examples/train_cnn_mnist.py) (same architecture, same disjoint train/test split). Run the one-time IDX export first (reuses BasicML's existing `scikit-learn` dependency to fetch/cache real MNIST, then writes it out as IDX files with no Python dependency at runtime for the C++ side):

```bash
python AdvanceML/scripts/prepare_mnist_idx.py     # writes AdvanceML/data/mnist/*, gitignored
cmake --build build --target train_cnn_mnist
./build/train_cnn_mnist AdvanceML/data/mnist
```

### Performance notes

On this machine (i7-1165G7, 4 cores / 8 threads), the full 10-epoch run takes about 18 s. Before TODO-0045 it took 75 s, and before the TODO-0043/0044 work several minutes. See `ai-audit/instructions/TODO-0045.md` for the profiles.

- oneDNN primitives are cached per op, shape and layout, so each is built once.
- `conv2d`, `batch_norm2d`, `relu`, `max_pool2d`/`avg_pool2d` and the global pools run on oneDNN and keep oneDNN's blocked layout (for example `nChw16c`) between them. A conv → BN → ReLU → pool chain never converts back to NCHW. The conversion happens only when something reads `tensor.data()`, or when an op without oneDNN support consumes the tensor.
- Tensor buffers (`FloatBuffer`) are 64-byte aligned and are not zero-filled when an op is about to overwrite them.

## Autograd

- `tensor.data()` is read-only. Write through `tensor.mutable_data()`: it bumps the storage's version counter, and `backward()` throws if a tensor it saved was written to after it was saved, so the gradient can't be silently wrong.
- Wrap evaluation in `NoGradGuard no_grad;`. Ops then record no graph, keep no saved tensors, and oneDNN primitives run in `forward_inference` mode.
- `loss.backward()` frees each node's saved tensors as it goes. Pass `loss.backward(/*retain_graph=*/true)` to backpropagate through the same graph again.
- Gradients are only computed for inputs that need them (for example, a first conv layer never computes a gradient for its data batch).
- `flatten`, `identity` and inference-mode `dropout` return views that share their input's storage.
- `tensor.data()` / `mutable_data()` return `const FloatBuffer&` / `FloatBuffer&` (a `std::vector<float>` with an aligned allocator). `FloatBuffer(n)` leaves its elements uninitialized, so use `FloatBuffer(n, 0.0f)` when you need zeros. `Tensor` still accepts a plain `std::vector<float>`, which it copies.
- A oneDNN-backed op may return a tensor in a blocked layout (`tensor.has_plain_layout() == false`). The first `data()` read converts it to NCHW in place. That read is not a write, so the version counter doesn't change.
- `Tensor` and the primitive caches are not thread-safe.
- Double backward (gradients of gradients) is not supported.

## Status

Component library at rough parity with BasicML: autograd engine, `Module`/`Sequential`/`Linear`/`Conv2D`/`MaxPool2D`/`AvgPool2D`/`Flatten`/`BatchNorm2D`, activations (ReLU/Sigmoid/LeakyReLU/GELU/Softmax), losses (`MSELoss`/`CrossEntropyLoss`/`SoftmaxCrossEntropyLoss`), optimizers (`SGD`/`Momentum`/`Adam`/`AdamW`), an MNIST IDX loader, an `Accuracy` metric, and the `train_cnn_mnist` example above (`ai-audit` TODO-0034 umbrella).
