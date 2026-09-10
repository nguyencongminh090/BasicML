# AI generated (refactored/authored with Claude Code)
"""Tiny CNN on a synthetic blob-image dataset.

Run directly: python BasicML/examples/train_cnn.py

Each image holds one Gaussian blob whose quadrant sets the label (see
:func:`basicml.datasets.make_blob_images`). A ``Conv2D -> ReLU -> MaxPool2D ->
Flatten -> Linear -> Sigmoid`` model is trained with binary cross-entropy and
Adam over mini-batches, driving the manual backward chain
``loss.backward() -> model.backward(grad)``. Falling loss and rising accuracy
are the check that the hand-derived convolution/pooling gradients are correct.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from basicml.nn.models    import CNNModel
from basicml.nn.loss      import BinaryCrossEntropy
from basicml.optim.adam   import Adam
from basicml.datasets     import make_blob_images, iter_minibatches
from basicml.metrics      import Accuracy
from basicml.visualize    import ProgressPrinter

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG -------------------------------------------------------------------
SEED          = 0
N_SAMPLES     = 240
IMAGE_SIZE    = 12
BLOB_SIGMA    = 1.3
NOISE         = 0.4
VAL_FRACTION  = 0.25

CONV_CHANNELS = 6
KERNEL_SIZE   = 3
POOL_SIZE     = 2

EPOCHS        = 40
BATCH_SIZE    = 32
LEARN_RATE    = 0.01

SHOW_PLOT     = True
# ---------------------------------------------------------------------------


def split_train_val(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images and labels into a training and a validation set.

    Args:
        X: Image array of shape ``(n_samples, 1, size, size)``.
        y: Binary label array of shape ``(n_samples, 1)``.

    Returns:
        Tuple ``(X_train, y_train, X_val, y_val)``.
    """
    n_val = int(len(X) * VAL_FRACTION)
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray, y_val: np.ndarray) -> CNNModel:
    """Fit the CNN with binary cross-entropy over mini-batches.

    Args:
        X_train: Training images, shape ``(n, 1, size, size)``.
        y_train: Training labels, shape ``(n, 1)``.
        X_val: Validation images.
        y_val: Validation labels.

    Returns:
        The trained :class:`~basicml.nn.models.CNNModel`.
    """
    model     = CNNModel(input_shape=(1, IMAGE_SIZE, IMAGE_SIZE),
                         conv_channels=CONV_CHANNELS,
                         kernel_size=KERNEL_SIZE,
                         pool_size=POOL_SIZE)
    criterion = BinaryCrossEntropy()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)
    progress  = ProgressPrinter({"acc": Accuracy()}, every=5)

    for epoch in range(1, EPOCHS + 1):
        for xb, yb in iter_minibatches(X_train, y_train, BATCH_SIZE, random_state=SEED + epoch):
            y_pred = model(xb)
            cost   = criterion(y_pred, yb)

            model.backward(criterion.backward())
            optimizer.step()
            optimizer.zero_grad()

            progress.update(y_pred, yb, cost)
        progress.end_epoch(epoch, EPOCHS)

    val_acc = Accuracy()(model(X_val), y_val)
    print(f"validation accuracy: {val_acc:.4f}")
    return model


def plot_samples(model: CNNModel, X: np.ndarray, y: np.ndarray) -> None:
    """Show a grid of validation images with predicted probabilities.

    Args:
        model: The trained model.
        X: Validation images, shape ``(n, 1, size, size)``.
        y: Validation labels, shape ``(n, 1)``.
    """
    probs      = model(X).reshape(-1)
    n_show     = min(12, len(X))
    fig, axes  = plt.subplots(3, 4, figsize=(8, 6))
    for ax, k in zip(axes.ravel(), range(n_show)):
        ax.imshow(X[k, 0], cmap="magma")
        ax.set_title(f"y={int(y[k, 0])}  p={probs[k]:.2f}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[n_show:]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def main() -> None:
    """Generate the dataset, train the CNN, and optionally plot predictions."""
    X, y = make_blob_images(n_samples=N_SAMPLES, size=IMAGE_SIZE,
                            blob_sigma=BLOB_SIGMA, noise=NOISE, random_state=SEED)
    X_train, y_train, X_val, y_val = split_train_val(X, y)

    model = train(X_train, y_train, X_val, y_val)
    if SHOW_PLOT:
        plot_samples(model, X_val, y_val)


if __name__ == "__main__":
    main()
