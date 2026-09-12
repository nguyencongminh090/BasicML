# AI generated (refactored/authored with Claude Code)
"""Deeper double-conv-per-block CNN on real MNIST, AdamW vs. Muon.

Run directly: python BasicML/examples/train_cnn_mnist_muon.py

Follow-up to ``train_cnn_mnist.py`` (single conv per block, AdamW, 96.84%
held-out test accuracy). Two changes, applied on the SAME train/test split
so the comparison is fair:

1. Architecture: the single-conv-per-block pattern is replaced with a
   double-conv-per-block pattern -- ``(Conv->BN->ReLU) x2 -> Pool ->
   Dropout`` per stage -- the single biggest lever in well-known
   high-accuracy MNIST CNN recipes (e.g. the "C3:32-C3:32-P2-C3:64-C3:64-P2"
   shape popularized by Chris Deotte's 99.5%-MNIST notebook). Two 3x3 convs
   before each downsample let the network build richer features at each
   spatial resolution before losing information to pooling; Dropout after
   each pool regularizes the extra capacity. ``GlobalAvgPool2D`` is kept as
   the classifier head (as in ``train_cnn_mnist.py``) to keep parameter
   count low despite the added depth.
2. Optimizer: this repo's ``Muon`` (added by the user this session) is the
   real Muon design -- momentum + Newton-Schulz orthogonalization -- which
   is meant to run only on 2D+ *hidden weight matrices*, not on biases or
   normalization scale/shift parameters (orthogonalizing a bias vector or a
   per-channel BatchNorm gamma/beta is not what the method is for, even
   though ``Muon.step()`` would technically accept them since they satisfy
   ``ndim >= 2`` after the ``(1, C, 1, 1)`` BatchNorm2D affine shape). So
   this script splits parameters into two groups -- ``Conv2D.w``/
   ``Linear.w`` go to ``Muon``, everything else (every bias and every
   ``BatchNorm2D`` gamma/beta) goes to a plain ``AdamW`` -- and trains the
   identical architecture once with AdamW alone (baseline) and once with
   this Muon+AdamW hybrid, from the same initial weights, for a clean
   optimizer-only comparison.
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
from basicml.nn.dropout    import Dropout
from basicml.nn.activation import ReLU, Softmax
from basicml.nn.loss       import CrossEntropyLoss
from basicml.optim.adamw   import AdamW
from basicml.optim.muon    import Muon
from basicml.datasets      import iter_minibatches
from basicml.metrics       import Accuracy

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ----------------------------------------------------------------
SEED         = 0
IMAGE_SIZE   = 28
# The double-conv-per-block pattern below runs ~4.7x slower per sample-epoch
# than train_cnn_mnist.py's single-conv blocks (measured, not estimated --
# an earlier "~2x" guess was wrong): 6 BatchNorm2D-backed conv layers
# instead of 3, plus 2 Dropout layers. N_TRAIN is reduced from that script's
# 15000 (to ~28 min/run instead of ~50+ min/run); N_TEST is kept identical
# so held-out accuracy stays comparable across both scripts.
N_TRAIN      = 10000
N_TEST       = 2500

# Double-conv-per-block channel pattern: C1a/C1b -> pool -> C2a/C2b -> pool
# -> C3a/C3b -> GAP. Kept at the same overall scale as train_cnn_mnist.py's
# single-conv stages (16/32/64) rather than the larger 32/64/128 some
# published recipes use, to keep this pure-numpy tap-loop run tractable.
STAGE1_CH    = 16
STAGE2_CH    = 32
STAGE3_CH    = 64
DROPOUT_P    = 0.25

EPOCHS       = 8
BATCH_SIZE   = 128
ADAMW_LR     = 0.001
MUON_LR      = 0.02
AUX_LR       = 0.001    # AdamW lr for the Muon run's non-weight parameters
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
    """Build the double-conv-per-block classifier as a dict of named ``basicml.nn`` layers.

    Args:
        seed: Seed for the ``Conv2D``/``Linear`` weight initialization.

    Returns:
        Dict mapping layer name to layer instance, in forward order.
    """
    np.random.seed(seed)
    return {
        "conv1a": Conv2D(1, STAGE1_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn1a"  : BatchNorm2D(STAGE1_CH),
        "relu1a": ReLU(),
        "conv1b": Conv2D(STAGE1_CH, STAGE1_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn1b"  : BatchNorm2D(STAGE1_CH),
        "relu1b": ReLU(),
        "pool1" : MaxPool2D(2),
        "drop1" : Dropout(DROPOUT_P),

        "conv2a": Conv2D(STAGE1_CH, STAGE2_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn2a"  : BatchNorm2D(STAGE2_CH),
        "relu2a": ReLU(),
        "conv2b": Conv2D(STAGE2_CH, STAGE2_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn2b"  : BatchNorm2D(STAGE2_CH),
        "relu2b": ReLU(),
        "pool2" : MaxPool2D(2),
        "drop2" : Dropout(DROPOUT_P),

        "conv3a": Conv2D(STAGE2_CH, STAGE3_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn3a"  : BatchNorm2D(STAGE3_CH),
        "relu3a": ReLU(),
        "conv3b": Conv2D(STAGE3_CH, STAGE3_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "bn3b"  : BatchNorm2D(STAGE3_CH),
        "relu3b": ReLU(),

        "gap"    : GlobalAvgPool2D(),
        "flatten": Flatten(),
        "linear" : Linear(STAGE3_CH, 10, init_type="xavier"),
        "softmax": Softmax(axis=-1),
    }


CONV_LAYER_NAMES   = ("conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b")
BATCHNORM_LAYER_NAMES = ("bn1a", "bn1b", "bn2a", "bn2b", "bn3a", "bn3b")


def weight_parameters(layers: dict[str, Module]) -> list:
    """Collect the 2D+ hidden weight matrices Muon is meant to optimize.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        Every ``Conv2D``/``Linear`` weight tensor (excludes biases).
    """
    params = [layers[name].w for name in CONV_LAYER_NAMES]     # type: ignore[attr-defined]
    params.append(layers["linear"].w)                          # type: ignore[attr-defined]
    return params


def aux_parameters(layers: dict[str, Module]) -> list:
    """Collect every parameter Muon should NOT touch: biases and BatchNorm affine params.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        Every ``Conv2D``/``Linear`` bias plus every ``BatchNorm2D``
        gamma/beta.
    """
    params = []
    for name in CONV_LAYER_NAMES:
        b = layers[name].b   # type: ignore[attr-defined]
        if b is not None:
            params.append(b)
    for name in BATCHNORM_LAYER_NAMES:
        params.extend(layers[name].parameters())
    linear_b = layers["linear"].b   # type: ignore[attr-defined]
    if linear_b is not None:
        params.append(linear_b)
    return params


def all_parameters(layers: dict[str, Module]) -> list:
    """Every trainable parameter, for the single-optimizer (AdamW-only) baseline.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        ``weight_parameters(layers) + aux_parameters(layers)``.
    """
    return weight_parameters(layers) + aux_parameters(layers)


def set_training(layers: dict[str, Module], mode: bool) -> None:
    """Toggle every layer's ``training`` flag (``Dropout``/``BatchNorm2D`` react to it).

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        mode: ``True`` for training, ``False`` for eval.
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
    out = X
    for name in ("conv1a", "bn1a", "relu1a", "conv1b", "bn1b", "relu1b", "pool1", "drop1",
                "conv2a", "bn2a", "relu2a", "conv2b", "bn2b", "relu2b", "pool2", "drop2",
                "conv3a", "bn3a", "relu3a", "conv3b", "bn3b", "relu3b",
                "gap", "flatten", "linear", "softmax"):
        out = layers[name](out)
    return out


_BACKWARD_ORDER = ("conv1a", "bn1a", "relu1a", "conv1b", "bn1b", "relu1b", "pool1", "drop1",
                   "conv2a", "bn2a", "relu2a", "conv2b", "bn2b", "relu2b", "pool2", "drop2",
                   "conv3a", "bn3a", "relu3a", "conv3b", "bn3b", "relu3b",
                   "gap", "flatten", "linear", "softmax")


def backward(layers: dict[str, Module], grad: np.ndarray) -> None:
    """Manually chain ``backward()`` through every layer, in reverse order.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        grad: Upstream gradient from the loss, shape matching the model's
            output.
    """
    for name in reversed(_BACKWARD_ORDER):
        grad = layers[name].backward(grad)


def evaluate(layers: dict[str, Module], criterion: CrossEntropyLoss,
            X: np.ndarray, y_onehot: np.ndarray, batch_size: int = 256) -> tuple[float, float]:
    """Compute mean loss and accuracy over a dataset, in eval mode.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        criterion: Loss used to score each batch.
        X: Images to evaluate, shape ``(N, 1, 28, 28)``.
        y_onehot: One-hot labels, shape ``(N, 10)``.
        batch_size: Batch size for the forward passes.

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


def train(layers: dict[str, Module], optimizers: list, X_train: np.ndarray, y_train_onehot: np.ndarray,
          X_test: np.ndarray, y_test_onehot: np.ndarray, label: str) -> list[EpochLog]:
    """Train with one or more optimizers stepped together, tracking metrics per epoch.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        optimizers: One optimizer (AdamW-only baseline) or two (Muon for
            weights + AdamW for everything else) -- all are stepped and
            zeroed every batch.
        X_train: Training images, shape ``(N_TRAIN, 1, 28, 28)``.
        y_train_onehot: One-hot training labels.
        X_test: Held-out test images, shape ``(N_TEST, 1, 28, 28)``.
        y_test_onehot: One-hot test labels.
        label: Run name, used only for the printed progress lines.

    Returns:
        One :class:`EpochLog` per epoch, in training order.
    """
    criterion = CrossEntropyLoss()
    logs      = []

    print(f"Training [{label}]...")
    for epoch in range(1, EPOCHS + 1):
        train_accuracy   = Accuracy(num_classes=10)
        train_loss_total = 0.0
        n_batches        = 0
        for xb, yb in iter_minibatches(X_train, y_train_onehot, BATCH_SIZE, random_state=SEED + epoch):
            probs = forward(layers, xb)
            cost  = criterion(probs, yb)

            backward(layers, criterion.backward())
            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad()

            train_accuracy.update(probs, yb)
            train_loss_total += cost
            n_batches        += 1

        train_loss = train_loss_total / n_batches
        train_acc  = float(train_accuracy.compute())
        test_loss, test_acc = evaluate(layers, criterion, X_test, y_test_onehot)
        logs.append(EpochLog(epoch, train_loss, train_acc, test_loss, test_acc))
        print(f"[{label}] epoch {epoch:>2}/{EPOCHS}  train_loss={train_loss:.4f}  "
             f"train_acc={train_acc:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    return logs


def plot_comparison(logs_adamw: list[EpochLog], logs_muon: list[EpochLog]) -> None:
    """Plot test loss/accuracy curves for the AdamW-only vs. Muon+AdamW runs.

    Args:
        logs_adamw: Per-epoch metrics from the AdamW-only baseline.
        logs_muon: Per-epoch metrics from the Muon(weights)+AdamW(aux) run.
    """
    epochs = [log.epoch for log in logs_adamw]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Double-Conv MNIST CNN: AdamW vs Muon")

    ax_loss.plot(epochs, [log.test_loss for log in logs_adamw], marker="o", label="AdamW")
    ax_loss.plot(epochs, [log.test_loss for log in logs_muon], marker="o", label="Muon+AdamW")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("test loss")
    ax_loss.set_title("Held-out test loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [log.test_acc for log in logs_adamw], marker="o", label="AdamW")
    ax_acc.plot(epochs, [log.test_acc for log in logs_muon], marker="o", label="Muon+AdamW")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("test accuracy")
    ax_acc.set_title("Held-out test accuracy"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    """Fetch MNIST once, then train the same architecture with AdamW and with Muon+AdamW."""
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = load_mnist_split(N_TRAIN, N_TEST, SEED)

    layers_adamw    = build_model(SEED)
    optimizer_adamw = AdamW(all_parameters(layers_adamw), lr=ADAMW_LR, weight_decay=WEIGHT_DECAY)
    logs_adamw      = train(layers_adamw, [optimizer_adamw], X_train, y_train_onehot,
                            X_test, y_test_onehot, label="AdamW")

    layers_muon  = build_model(SEED)   # identical init: np.random.seed(SEED) inside build_model
    muon         = Muon(weight_parameters(layers_muon), lr=MUON_LR, momentum=0.95)
    aux_adamw    = AdamW(aux_parameters(layers_muon), lr=AUX_LR, weight_decay=WEIGHT_DECAY)
    logs_muon    = train(layers_muon, [muon, aux_adamw], X_train, y_train_onehot,
                        X_test, y_test_onehot, label="Muon+AdamW")

    print(f"\nFinal test accuracy -- AdamW: {logs_adamw[-1].test_acc:.4f}  "
         f"Muon+AdamW: {logs_muon[-1].test_acc:.4f}")
    if SHOW_PLOT:
        plot_comparison(logs_adamw, logs_muon)


if __name__ == "__main__":
    main()
