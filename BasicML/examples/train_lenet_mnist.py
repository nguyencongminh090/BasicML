# AI generated (refactored/authored with Claude Code)
"""LeNet-shaped CNN on real MNIST, with a genuine held-out test split, AdamW vs. Muon.

Run directly: python BasicML/examples/train_lenet_mnist.py

LeNet-5 was designed for handwritten digit/character recognition, and MNIST
is exactly the kind of small, low-variability problem it targets -- unlike
the double-conv architecture tried in ``train_cnn_mnist_muon.py`` (which
turned out to run ~4.7x slower per sample-epoch and got its comparison run
cancelled before finishing), this model is cheap: the same 6/16-channel,
120/84-FC shape used by ``demo/plot_dynamic_lenet_feature_maps.py``, just
with a real train/test split (rather than that demo's train-accuracy-only,
1200-image reporting) and two optimizer runs to compare fairly.

As in ``train_cnn_mnist_muon.py``, ``Muon`` (Newton-Schulz orthogonalized
momentum) is applied only to the 2D+ hidden weight matrices
(``Conv2D.w``/``Linear.w``); every bias stays on a plain ``AdamW`` alongside
it, matching how Muon is meant to be used. The same architecture is trained
once with AdamW alone (baseline) and once with the Muon+AdamW hybrid, from
identical initial weights, on the same split.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from sklearn.datasets import fetch_openml

from basicml.nn.module     import Module
from basicml.nn.conv       import Conv2D
from basicml.nn.pool       import MaxPool2D
from basicml.nn.flatten    import Flatten
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Softmax
from basicml.nn.loss       import CrossEntropyLoss
from basicml.optim.optimizer import Optimizer
from basicml.optim.adamw   import AdamW
from basicml.optim.muon    import Muon
from basicml.datasets      import iter_minibatches
from basicml.metrics       import Accuracy

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ----------------------------------------------------------------
SEED       = 0
IMAGE_SIZE = 28
N_TRAIN    = 15000
N_TEST     = 2500

CONV1_CH   = 6      # LeNet-5's original C1 channel count
CONV2_CH   = 16     # LeNet-5's original C3 channel count
FC1_UNITS  = 120    # LeNet-5's original F5
FC2_UNITS  = 84     # LeNet-5's original F6

EPOCHS       = 15
BATCH_SIZE   = 64
ADAMW_LR     = 0.001
MUON_LR      = 0.02
AUX_LR       = 0.001    # AdamW lr for the Muon run's bias parameters
WEIGHT_DECAY = 0.01

SHOW_PLOT = True
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
    """Build the LeNet-shaped classifier as a dict of named ``basicml.nn`` layers.

    Args:
        seed: Seed for the ``Conv2D``/``Linear`` weight initialization.

    Returns:
        Dict mapping layer name to layer instance, in forward order.
    """
    np.random.seed(seed)
    conv1_out_size = IMAGE_SIZE - 5 + 1
    pool1_out_size = conv1_out_size // 2
    conv2_out_size = pool1_out_size - 5 + 1
    pool2_out_size = conv2_out_size // 2
    flat_features  = CONV2_CH * pool2_out_size ** 2

    return {
        "conv1"  : Conv2D(1, CONV1_CH, kernel_size=5, stride=1, padding=0, init_type="he"),
        "relu1"  : ReLU(),
        "pool1"  : MaxPool2D(2),
        "conv2"  : Conv2D(CONV1_CH, CONV2_CH, kernel_size=5, stride=1, padding=0, init_type="he"),
        "relu2"  : ReLU(),
        "pool2"  : MaxPool2D(2),
        "flatten": Flatten(),
        "fc1"    : Linear(flat_features, FC1_UNITS, init_type="he"),
        "relu3"  : ReLU(),
        "fc2"    : Linear(FC1_UNITS, FC2_UNITS, init_type="he"),
        "relu4"  : ReLU(),
        "fc3"    : Linear(FC2_UNITS, 10, init_type="xavier"),
        "softmax": Softmax(axis=-1),
    }


WEIGHT_LAYER_NAMES = ("conv1", "conv2", "fc1", "fc2", "fc3")


def weight_parameters(layers: dict[str, Module]) -> list:
    """Collect the 2D+ hidden weight matrices Muon is meant to optimize.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        Every ``Conv2D``/``Linear`` weight tensor (excludes biases).
    """
    return [layers[name].w for name in WEIGHT_LAYER_NAMES]   # type: ignore[attr-defined]


def aux_parameters(layers: dict[str, Module]) -> list:
    """Collect every bias -- the parameters Muon should NOT touch.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        Every ``Conv2D``/``Linear`` bias.
    """
    params = []
    for name in WEIGHT_LAYER_NAMES:
        b = layers[name].b   # type: ignore[attr-defined]
        if b is not None:
            params.append(b)
    return params


def all_parameters(layers: dict[str, Module]) -> list:
    """Every trainable parameter, for the single-optimizer (AdamW-only) baseline.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        ``weight_parameters(layers) + aux_parameters(layers)``.
    """
    return weight_parameters(layers) + aux_parameters(layers)


def forward(layers: dict[str, Module], X: np.ndarray) -> np.ndarray:
    """Run ``X`` through every layer in order.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        X: Input batch, shape ``(N, 1, 28, 28)``.

    Returns:
        Softmax class probabilities, shape ``(N, 10)``.
    """
    out = X
    for name in ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "flatten",
                "fc1", "relu3", "fc2", "relu4", "fc3", "softmax"):
        out = layers[name](out)
    return out


_BACKWARD_ORDER = ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "flatten",
                   "fc1", "relu3", "fc2", "relu4", "fc3", "softmax")


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
    """Compute mean loss and accuracy over a dataset.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        criterion: Loss used to score each batch.
        X: Images to evaluate, shape ``(N, 1, 28, 28)``.
        y_onehot: One-hot labels, shape ``(N, 10)``.
        batch_size: Batch size for the forward passes.

    Returns:
        Tuple ``(mean_loss, accuracy)``.
    """
    accuracy   = Accuracy(num_classes=10)
    total_loss = 0.0
    n_batches  = 0
    for xb, yb in iter_minibatches(X, y_onehot, batch_size, shuffle=False):
        probs = forward(layers, xb)
        total_loss += criterion(probs, yb)
        accuracy.update(probs, yb)
        n_batches += 1
    return total_loss / n_batches, float(accuracy.compute())


def cosine_decay_lr(base_lr: float, epoch: int, total_epochs: int, min_lr_ratio: float = 0.05) -> float:
    """Cosine-decay a learning rate from ``base_lr`` down to ``base_lr * min_lr_ratio``.

    Muon's earlier full run (FIX-0047) diverged from roughly epoch 8 onward
    with a fixed ``lr=0.02`` -- stable early, then the loss climbs every
    epoch after, the textbook signature of a learning rate that stays too
    high once training has gotten close to a good solution. Decaying it
    smoothly over the run is the standard fix (this is also how Muon is
    used in practice, e.g. the nanoGPT speedrun's warmup-stable-decay
    schedule), rather than picking one fixed, smaller learning rate that
    would converge slower everywhere just to stay stable at the end.

    Args:
        base_lr: Learning rate at ``epoch == 1``.
        epoch: Current epoch, 1-indexed.
        total_epochs: Total number of epochs in the run.
        min_lr_ratio: Floor as a fraction of ``base_lr``, reached at the
            final epoch.

    Returns:
        The learning rate to use for this epoch.
    """
    progress = (epoch - 1) / max(1, total_epochs - 1)
    factor   = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
    return base_lr * factor


def train(layers: dict[str, Module], optimizers: list, X_train: np.ndarray, y_train_onehot: np.ndarray,
          X_test: np.ndarray, y_test_onehot: np.ndarray, label: str,
          decay: Optional[tuple[Optimizer, float]] = None) -> list[EpochLog]:
    """Train with one or more optimizers stepped together, tracking metrics per epoch.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        optimizers: One optimizer (AdamW-only baseline) or two (Muon for
            weights + AdamW for biases) -- all are stepped and zeroed every
            batch.
        X_train: Training images, shape ``(N_TRAIN, 1, 28, 28)``.
        y_train_onehot: One-hot training labels.
        X_test: Held-out test images, shape ``(N_TEST, 1, 28, 28)``.
        y_test_onehot: One-hot test labels.
        label: Run name, used only for the printed progress lines.
        decay: Optional ``(optimizer, base_lr)`` pair -- if given, that
            optimizer's ``lr`` is cosine-decayed every epoch via
            :func:`cosine_decay_lr`. Every other optimizer in
            ``optimizers`` keeps a fixed ``lr``.

    Returns:
        One :class:`EpochLog` per epoch, in training order.
    """
    criterion = CrossEntropyLoss()
    logs      = []

    print(f"Training [{label}]...")
    for epoch in range(1, EPOCHS + 1):
        if decay is not None:
            decayed_optimizer, base_lr = decay
            decayed_optimizer.lr = cosine_decay_lr(base_lr, epoch, EPOCHS)

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
        logs_muon: Per-epoch metrics from the Muon(weights)+AdamW(biases) run.
    """
    epochs = [log.epoch for log in logs_adamw]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - LeNet on MNIST: AdamW vs Muon")

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
    """Fetch MNIST once, then train the same LeNet-shaped model with AdamW and with Muon+AdamW."""
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = load_mnist_split(N_TRAIN, N_TEST, SEED)

    layers_adamw    = build_model(SEED)
    optimizer_adamw = AdamW(all_parameters(layers_adamw), lr=ADAMW_LR, weight_decay=WEIGHT_DECAY)
    logs_adamw      = train(layers_adamw, [optimizer_adamw], X_train, y_train_onehot,
                            X_test, y_test_onehot, label="AdamW")

    layers_muon = build_model(SEED)   # identical init: np.random.seed(SEED) inside build_model
    muon        = Muon(weight_parameters(layers_muon), lr=MUON_LR, momentum=0.95)
    aux_adamw   = AdamW(aux_parameters(layers_muon), lr=AUX_LR, weight_decay=WEIGHT_DECAY)
    logs_muon   = train(layers_muon, [muon, aux_adamw], X_train, y_train_onehot,
                        X_test, y_test_onehot, label="Muon+AdamW",
                        decay=(muon, MUON_LR))

    print(f"\nFinal test accuracy -- AdamW: {logs_adamw[-1].test_acc:.4f}  "
         f"Muon+AdamW: {logs_muon[-1].test_acc:.4f}")
    if SHOW_PLOT:
        plot_comparison(logs_adamw, logs_muon)


if __name__ == "__main__":
    main()
