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

`train_cnn_mnist` — a `(Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D) x2 -> Conv2D -> BatchNorm2D -> ReLU -> AvgPool2D -> Flatten -> Linear -> Softmax` CNN trained on real MNIST with `AdamW`, the AdvanceML counterpart to [`BasicML/examples/train_cnn_mnist.py`](../BasicML/examples/train_cnn_mnist.py) (same architecture, same disjoint train/test split). Run the one-time IDX export first (reuses BasicML's existing `scikit-learn` dependency to fetch/cache real MNIST, then writes it out as IDX files with no Python dependency at runtime for the C++ side):

```bash
python AdvanceML/scripts/prepare_mnist_idx.py     # writes AdvanceML/data/mnist/*, gitignored
cmake --build build --target train_cnn_mnist
./build/train_cnn_mnist AdvanceML/data/mnist
```

### Known limitation: per-call oneDNN primitive creation

`conv2d`/`max_pool2d`/`avg_pool2d`/`batch_norm2d` each build a fresh oneDNN primitive descriptor on every forward/backward call rather than caching it across calls with the same shape — the straightforward-but-unoptimized approach taken while landing TODO-0038. At the small batch sizes (128) and channel counts this example uses, that per-call overhead currently dominates: a rough timed comparison against `BasicML/examples/train_cnn_mnist.py` at a reduced scale (2000 train / 400 test images, 3 epochs, otherwise identical config) measured AdvanceML at ~117s wall-clock vs. BasicML's ~32s — AdvanceML currently slower despite burning far more CPU-seconds across threads, the opposite of the CPU-throughput goal stated in `ai-audit/instructions/TODO-0030.md`. Primitive-descriptor caching (keyed by shape) is the natural follow-up to actually realize that goal and is not yet filed as its own TODO.

## Autograd

- `tensor.data()` is read-only. Write through `tensor.mutable_data()`: it bumps the storage's version counter, and `backward()` throws if a tensor it saved was written to after it was saved, so the gradient can't be silently wrong.
- Wrap evaluation in `NoGradGuard no_grad;`. Ops then record no graph, keep no saved tensors, and oneDNN primitives run in `forward_inference` mode.
- `loss.backward()` frees each node's saved tensors as it goes. Pass `loss.backward(/*retain_graph=*/true)` to backpropagate through the same graph again.
- Gradients are only computed for inputs that need them (for example, a first conv layer never computes a gradient for its data batch).
- `flatten`, `identity` and inference-mode `dropout` return views that share their input's storage.
- Double backward (gradients of gradients) is not supported.

## Status

Component library at rough parity with BasicML: autograd engine, `Module`/`Sequential`/`Linear`/`Conv2D`/`MaxPool2D`/`AvgPool2D`/`Flatten`/`BatchNorm2D`, activations (ReLU/Sigmoid/LeakyReLU/GELU/Softmax), losses (`MSELoss`/`CrossEntropyLoss`), optimizers (`SGD`/`Momentum`/`Adam`/`AdamW`), an MNIST IDX loader, an `Accuracy` metric, and the `train_cnn_mnist` example above (`ai-audit` TODO-0034 umbrella). See the known-limitation note above for the current CPU-throughput gap.
