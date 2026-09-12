# AI generated (refactored/authored with Claude Code)
"""Efficient CNN on real MNIST, with a genuine held-out test split.

Run directly: python BasicML/examples/train_cnn_mnist.py

Follow-up to this repo's ``demo/plot_dynamic_*_feature_maps.py`` family
(custom CNN / LeNet-shaped / AlexNet-shaped): those three report *train*
accuracy on a 1200-3000 image subset with no held-out test set, which isn't
a trustworthy accuracy number regardless of architecture. This script fixes
that -- a real train/test split out of MNIST's 70000 images -- and pairs it
with a more parameter-efficient architecture than any of the three demos:

``(Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D) x2 -> Conv2D -> BatchNorm2D ->
ReLU -> GlobalAvgPool2D -> Linear -> Softmax``

``BatchNorm2D`` (new in this session, added to ``basicml.nn.batchnorm``
alongside the pre-existing 2D-only ``BatchNorm``) normalizes each channel's
activations over the batch, stabilizing and speeding up training.
``GlobalAvgPool2D`` (already in ``basicml.nn.pool``, unused by any demo so
far) collapses each channel to one number instead of flattening the full
spatial map into a large ``Linear`` layer -- far fewer parameters than the
LeNet/AlexNet demos' 120-512-unit FC layers, and a standard modern-CNN
efficiency trick. Training uses ``AdamW`` (already in ``basicml.optim``,
unused by any demo so far) instead of plain ``Adam``, for its decoupled
weight decay.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.datasets import fetch_openml

from basicml.nn.module     import Module
from basicml.nn.conv       import Conv2D
from basicml.nn.pool       import MaxPool2D, GlobalAvgPool2D
from basicml.nn.batchnorm  import BatchNorm2D
from basicml.nn.flatten    import Flatten
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Softmax
from basicml.nn.loss       import CrossEntropyLoss
from basicml.optim.adamw   import AdamW
from basicml.datasets      import iter_minibatches
from basicml.metrics       import Accuracy

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ----------------------------------------------------------------
SEED         = 0
IMAGE_SIZE   = 28
N_TRAIN      = 15000
N_TEST       = 2500

CONV1_CH     = 16
CONV2_CH     = 32
CONV3_CH     = 64

EPOCHS       = 10
BATCH_SIZE   = 128
LEARN_RATE   = 0.001
WEIGHT_DECAY = 0.01

SHOW_PLOT    = True
# ---------------------------------------------------------------------------


@dataclass
class EpochLog:
    """Train/test metrics recorded after one epoch.

    Attributes:
        epoch: Epoch number (1-indexed).
        train_loss: Mean training cross-entropy loss over the epoch.
        train_acc: Training accuracy over the epoch.
        test_loss: Cross-entropy loss on the held-out test set.
        test_acc: Accuracy on the held-out test set.
    """
    epoch     : int
    train_loss: float
    train_acc : float
    test_loss : float
    test_acc  : float


def load_mnist_split(n_train: int, n_test: int, random_state: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch real MNIST via scikit-learn and split into disjoint train/test sets.

    Downloads once (``fetch_openml`` caches to disk afterward), normalizes
    pixels to ``[0, 1]``, and one-hot encodes the digit labels. Train and
    test indices are drawn from the same shuffled permutation with no
    overlap, so test accuracy reflects genuinely unseen digits.

    Args:
        n_train: Number of images for the training set.
        n_test: Number of images for the held-out test set.
        random_state: Seed for the shuffling permutation.

    Returns:
        Tuple ``(X_train, y_train_int, y_train_onehot, X_test, y_test_int,
        y_test_onehot)``.
    """
    print("Fetching MNIST (cached after first download)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    rng     = np.random.RandomState(random_state)
    indices = rng.permutation(len(mnist.data))
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:n_train + n_test]

    def prepare(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = mnist.data[idx].astype(np.float64) / 255.0
        X = X.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        y_int    = mnist.target[idx].astype(np.int64)
        y_onehot = np.eye(10)[y_int]
        return X, y_int, y_onehot

    X_train, y_train_int, y_train_onehot = prepare(train_idx)
    X_test, y_test_int, y_test_onehot    = prepare(test_idx)
    return X_train, y_train_int, y_train_onehot, X_test, y_test_int, y_test_onehot


def build_model(seed: int) -> dict[str, Module]:
    """Build the efficient CNN classifier as a plain dict of named ``basicml.nn`` layers.

    Args:
        seed: Seed for the ``Conv2D``/``Linear`` weight initialization.

    Returns:
        Dict mapping layer name to layer instance, in forward order.
    """
    np.random.seed(seed)
    return {
        "conv1"  : Conv2D(1, CONV1_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn1"    : BatchNorm2D(CONV1_CH),
        "relu1"  : ReLU(),
        "pool1"  : MaxPool2D(2),
        "conv2"  : Conv2D(CONV1_CH, CONV2_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn2"    : BatchNorm2D(CONV2_CH),
        "relu2"  : ReLU(),
        "pool2"  : MaxPool2D(2),
        "conv3"  : Conv2D(CONV2_CH, CONV3_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn3"    : BatchNorm2D(CONV3_CH),
        "relu3"  : ReLU(),
        "gap"    : GlobalAvgPool2D(),
        "flatten": Flatten(),
        "linear" : Linear(CONV3_CH, 10, init_type="xavier"),
        "softmax": Softmax(axis=-1),
    }


def set_training(layers: dict[str, Module], mode: bool) -> None:
    """Toggle every layer's ``training`` flag (``BatchNorm2D`` reacts to it).

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        mode: ``True`` for training (batch statistics), ``False`` for eval
            (running statistics -- required for a fair test-set evaluation).
    """
    for layer in layers.values():
        layer.train(mode)


def forward(layers: dict[str, Module], X: np.ndarray) -> np.ndarray:
    """Run ``X`` through every layer in order.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        X: Input batch, shape ``(N, 1, 28, 28)``.

    Returns:
        Softmax class probabilities, shape ``(N, 10)``.
    """
    out = layers["conv1"](X)
    out = layers["bn1"](out)
    out = layers["relu1"](out)
    out = layers["pool1"](out)
    out = layers["conv2"](out)
    out = layers["bn2"](out)
    out = layers["relu2"](out)
    out = layers["pool2"](out)
    out = layers["conv3"](out)
    out = layers["bn3"](out)
    out = layers["relu3"](out)
    out = layers["gap"](out)
    out = layers["flatten"](out)
    out = layers["linear"](out)
    return layers["softmax"](out)


def backward(layers: dict[str, Module], grad: np.ndarray) -> None:
    """Manually chain ``backward()`` through every layer, in reverse order.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        grad: Upstream gradient from the loss, shape matching the model's
            output.
    """
    grad = layers["softmax"].backward(grad)
    grad = layers["linear"].backward(grad)
    grad = layers["flatten"].backward(grad)
    grad = layers["gap"].backward(grad)
    grad = layers["relu3"].backward(grad)
    grad = layers["bn3"].backward(grad)
    grad = layers["conv3"].backward(grad)
    grad = layers["pool2"].backward(grad)
    grad = layers["relu2"].backward(grad)
    grad = layers["bn2"].backward(grad)
    grad = layers["conv2"].backward(grad)
    grad = layers["pool1"].backward(grad)
    grad = layers["relu1"].backward(grad)
    grad = layers["bn1"].backward(grad)
    layers["conv1"].backward(grad)


def parameters(layers: dict[str, Module]) -> list:
    """Collect every trainable parameter across the model's layers.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        Flat list of ``Tensor`` parameters, in layer order.
    """
    params = []
    for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "linear"):
        params.extend(layers[name].parameters())
    return params


def evaluate(layers: dict[str, Module], criterion: CrossEntropyLoss,
            X: np.ndarray, y_onehot: np.ndarray, batch_size: int = 256) -> tuple[float, float]:
    """Compute mean loss and accuracy over a dataset, in eval mode.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        criterion: Loss used to score each batch.
        X: Images to evaluate, shape ``(N, 1, 28, 28)``.
        y_onehot: One-hot labels, shape ``(N, 10)``.
        batch_size: Batch size for the forward passes (no backward pass
            happens here, so this only bounds peak memory).

    Returns:
        Tuple ``(mean_loss, accuracy)``.
    """
    set_training(layers, False)
    accuracy   = Accuracy(num_classes=10)
    total_loss = 0.0
    n_batches  = 0
    for xb, yb in iter_minibatches(X, y_onehot, batch_size, shuffle=False):
        probs = forward(layers, xb)
        total_loss += criterion(probs, yb)
        accuracy.update(probs, yb)
        n_batches += 1
    set_training(layers, True)
    return total_loss / n_batches, float(accuracy.compute())


def train(layers: dict[str, Module], X_train: np.ndarray, y_train_onehot: np.ndarray,
          X_test: np.ndarray, y_test_onehot: np.ndarray) -> list[EpochLog]:
    """Train with AdamW while tracking train/test loss and accuracy per epoch.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        X_train: Training images, shape ``(N_TRAIN, 1, 28, 28)``.
        y_train_onehot: One-hot training labels.
        X_test: Held-out test images, shape ``(N_TEST, 1, 28, 28)``.
        y_test_onehot: One-hot test labels.

    Returns:
        One :class:`EpochLog` per epoch, in training order.
    """
    criterion = CrossEntropyLoss()
    optimizer = AdamW(parameters(layers), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)
    logs      = []

    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        train_accuracy   = Accuracy(num_classes=10)
        train_loss_total = 0.0
        n_batches        = 0
        for xb, yb in iter_minibatches(X_train, y_train_onehot, BATCH_SIZE, random_state=SEED + epoch):
            probs = forward(layers, xb)
            cost  = criterion(probs, yb)

            backward(layers, criterion.backward())
            optimizer.step()
            optimizer.zero_grad()

            train_accuracy.update(probs, yb)
            train_loss_total += cost
            n_batches        += 1

        # Running-batch metrics, not a second full pass over X_train: each
        # batch's loss/accuracy is measured while its own weights (before
        # that batch's update) were in effect, then averaged over the epoch --
        # cheaper than re-scoring the whole training set, and standard
        # practice for a "train metrics" curve.
        train_loss = train_loss_total / n_batches
        train_acc  = float(train_accuracy.compute())
        test_loss, test_acc = evaluate(layers, criterion, X_test, y_test_onehot)
        logs.append(EpochLog(epoch, train_loss, train_acc, test_loss, test_acc))
        print(f"epoch {epoch:>2}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
             f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    return logs


def plot_curves(logs: list[EpochLog]) -> None:
    """Plot train/test loss and accuracy curves across epochs.

    Args:
        logs: Per-epoch metrics, as returned by :func:`train`.
    """
    epochs     = [log.epoch for log in logs]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Efficient CNN on MNIST")

    ax_loss.plot(epochs, [log.train_loss for log in logs], marker="o", label="train")
    ax_loss.plot(epochs, [log.test_loss for log in logs], marker="o", label="test")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [log.train_acc for log in logs], marker="o", label="train")
    ax_acc.plot(epochs, [log.test_acc for log in logs], marker="o", label="test")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    """Fetch a real MNIST train/test split, train the efficient CNN, and report test accuracy."""
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = load_mnist_split(N_TRAIN, N_TEST, SEED)

    layers = build_model(SEED)
    logs   = train(layers, X_train, y_train_onehot, X_test, y_test_onehot)

    print(f"\nFinal held-out test accuracy: {logs[-1].test_acc:.4f} "
         f"on {N_TEST} unseen images (trained on {N_TRAIN})")
    if SHOW_PLOT:
        plot_curves(logs)


if __name__ == "__main__":
    main()
