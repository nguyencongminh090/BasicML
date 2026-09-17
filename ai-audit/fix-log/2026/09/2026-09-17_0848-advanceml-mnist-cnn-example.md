---
id: FIX-0061
timestamp: 2026-09-17T08:48:22+07:00
todo_id: TODO-0040
---

## Prompt

Continuing the same "DO Task-0034" session after merging TODO-0038/TODO-0039 into `dev` (FIX-0060): area 6 of the TODO-0034 umbrella, a runnable MNIST CNN example for AdvanceML analogous to `BasicML/examples/train_cnn_mnist.py`, plus a rough perf comparison against it.

## Action

- New `AdvanceML/examples/train_cnn_mnist.cpp` — same architecture as the BasicML script (`(Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D) x2 -> Conv2D -> BatchNorm2D -> ReLU -> AvgPool2D(7,7) -> Flatten -> Linear -> Softmax`, `AdamW`, `EPOCHS=10`/`BATCH_SIZE=128`/`lr=0.001`/`weight_decay=0.01`), built on TODO-0038's `Conv2D`/`BatchNorm2D`/`MaxPool2D`/`AvgPool2D`/`Flatten` and TODO-0039's `datasets::load_mnist`/`metrics::Accuracy`. `AvgPool2D(7, 7)` stands in for BasicML's `GlobalAvgPool2D` (AdvanceML has no dedicated global-pool layer). Wired into `AdvanceML/CMakeLists.txt` as its own `train_cnn_mnist` executable target, separate from the `advanceml_tests` Catch2 binary.
- New `AdvanceML/scripts/prepare_mnist_idx.py` — this machine has no direct access to the original `yann.lecun.com` IDX distribution, so this one-time script reuses BasicML's existing `scikit-learn` dependency (`fetch_openml`, already cached locally) to fetch the same MNIST data BasicML's examples use, re-splits it exactly like `BasicML/examples/train_cnn_mnist.py::load_mnist_split` (same seed/`N_TRAIN`/`N_TEST`), and writes IDX3/IDX1 ubyte files to `AdvanceML/data/mnist/` (gitignored, regenerable) for the from-scratch C++ loader to read. Keeps the C++ binary itself free of any Python dependency at runtime.
- Smoke-tested with synthetic random IDX fixtures, then with a real 1000/200-image MNIST subset, before running a real timed comparison.
- **Rough perf comparison** (2000 train / 400 test images, 3 epochs, otherwise identical config, each measured in isolation with no other heavy process competing for CPU): `BasicML/examples/train_cnn_mnist.py` (via a scratch copy with `N_TRAIN`/`N_TEST`/`EPOCHS` overridden and `SHOW_PLOT=False`) took ~32s wall-clock (~33s user CPU); AdvanceML's `train_cnn_mnist` took ~117s wall-clock (~13m37s user CPU across threads). AdvanceML is currently ~3.7x slower in wall-clock and burns roughly 25x more CPU-seconds for that result.
- `cmake --build AdvanceML/build` + `ctest --test-dir AdvanceML/build --output-on-failure`: 31/31 tests still passing (the new example target doesn't affect the test binary).
- Updated `AdvanceML/README.md`: replaced the stale "scaffold only" Status section with the current component-library summary, added an "Examples" section documenting `prepare_mnist_idx.py` + `train_cnn_mnist`, and a "Known limitation" note with the timed comparison above.
- Added `AdvanceML/data/` to `.gitignore` (generated dataset, like BasicML's `fetch_openml` cache).

## Decision

- Did **not** run the full `N_TRAIN=15000`/`N_TEST=2500`/`EPOCHS=10` comparison end to end: an initial attempt (with several other heavy processes competing for CPU at the same time) showed severe contention artifacts, and an isolated single-epoch timing at full scale measured ~300s/epoch (~50 min projected for all 10) — a large amount of session wall-clock for a "rough" comparison that wouldn't change the substantive finding already visible at the smaller scale. Reported the smaller, cleanly-isolated 2000/400/3-epoch numbers instead, and said so explicitly rather than presenting the isolated single full-scale epoch as if it were the complete picture.
- Root-caused (but did not fix) the slowdown: `conv2d`/`max_pool2d`/`avg_pool2d`/`batch_norm2d` (`AdvanceML/src/ops.cpp`) each build a fresh oneDNN primitive descriptor on every forward/backward call rather than caching by shape — the straightforward-but-unoptimized approach taken while landing TODO-0038, not something introduced here. At this example's batch/channel sizes that per-call descriptor-creation cost dominates actual compute, which is why AdvanceML is currently slower than BasicML's plain NumPy despite burning far more CPU across threads. Fixing this (primitive-descriptor caching keyed by shape) is a legitimate follow-up but out of scope for "produce a runnable example + rough comparison" — noted in the README rather than silently fixed or silently ignored, and not yet filed as its own TODO (left for whoever picks it up next to decide priority).
- Used `AvgPool2D(kernel_size=7, stride=7)` rather than adding a new `GlobalAvgPool2D` layer: at this architecture's fixed 28x28 input the post-pool2 spatial size is always exactly 7x7, so a literal `AvgPool2D` call is equivalent to a global pool here with zero new code, and BasicML already has a dedicated `GlobalAvgPool2D` for the general case if AdvanceML needs one later.
- Real MNIST via a one-time Python conversion script (not synthetic data, not a raw network fetch of the original ubyte files) — this machine has no reachable copy of the original `yann.lecun.com` distribution, but BasicML's `scikit-learn` dependency already has the same 70000-image dataset cached, so reusing it avoids adding a new data dependency while still exercising the loader against genuine images.

## Conclusion

Landed. Area 6 (the last open area of the TODO-0034 umbrella) is done: a runnable, tested MNIST CNN example exists for AdvanceML, and a real (if reduced-scale) perf comparison against BasicML's equivalent is documented, honestly showing AdvanceML currently *slower* at small scale due to uncached oneDNN primitive creation — the CPU-throughput goal from TODO-0030 is not yet met, and the README says so plainly rather than overstating the result. TODO-0034 itself can now be closed as all six areas are done or explicitly handed off (the primitive-caching optimization, noted but not filed as its own TODO).
