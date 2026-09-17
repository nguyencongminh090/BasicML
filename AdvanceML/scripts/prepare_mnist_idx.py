# AI generated (refactored/authored with Claude Code)
"""One-time MNIST fetch + IDX-format export for AdvanceML's C++ example.

Run directly: python AdvanceML/scripts/prepare_mnist_idx.py

AdvanceML/examples/train_cnn_mnist.cpp loads real MNIST through
`advanceml::datasets::load_mnist`, a from-scratch IDX-format parser with no
Python dependency at runtime (see TODO-0039) -- but something still has to
produce IDX files for it to read, since this machine has no direct access to
the original http://yann.lecun.com/exdb/mnist/ ubyte distribution. This
script is that one-time step: it reuses BasicML's existing `scikit-learn`
dependency (`sklearn.datasets.fetch_openml`, cached under
`~/scikit_learn_data` after the first run) to pull the same 70000-image
MNIST dataset BasicML's examples use, re-splits it exactly like
`BasicML/examples/train_cnn_mnist.py::load_mnist_split` (same seed, same
`N_TRAIN`/`N_TEST`), and writes the four resulting arrays out as classic
IDX3 (images) / IDX1 (labels) ubyte files under `AdvanceML/data/mnist/` --
the same on-disk format the C++ loader already knows how to parse, gitignored
like BasicML's fetch_openml cache since it's regenerable data, not source.
"""
import os
import struct
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import fetch_openml

IMAGE_SIZE = 28
N_TRAIN = 15000
N_TEST = 2500
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mnist")


def write_idx_images(path: str, images: np.ndarray) -> None:
    """Writes `images` (N, rows, cols) uint8 pixels as an IDX3 ubyte file."""
    n, rows, cols = images.shape
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(images.astype(np.uint8).tobytes())


def write_idx_labels(path: str, labels: np.ndarray) -> None:
    """Writes `labels` (N,) uint8 digit labels as an IDX1 ubyte file."""
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, labels.shape[0]))
        f.write(labels.astype(np.uint8).tobytes())


def main() -> None:
    """Fetches MNIST, splits it like BasicML's CNN example, and writes IDX files."""
    print("Fetching MNIST (cached after first download)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(mnist.data))
    train_idx = indices[:N_TRAIN]
    test_idx = indices[N_TRAIN:N_TRAIN + N_TEST]

    images = mnist.data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    labels = mnist.target.astype(np.uint8)

    os.makedirs(OUT_DIR, exist_ok=True)
    write_idx_images(os.path.join(OUT_DIR, "train-images-idx3-ubyte"), images[train_idx])
    write_idx_labels(os.path.join(OUT_DIR, "train-labels-idx1-ubyte"), labels[train_idx])
    write_idx_images(os.path.join(OUT_DIR, "test-images-idx3-ubyte"), images[test_idx])
    write_idx_labels(os.path.join(OUT_DIR, "test-labels-idx1-ubyte"), labels[test_idx])
    print(f"Wrote {N_TRAIN} train / {N_TEST} test images to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
