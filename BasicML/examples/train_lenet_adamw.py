# AI generated (refactored/authored with Claude Code)
"""LeNet-shaped CNN on real MNIST, trained with plain AdamW, held-out test split.

Run directly: python BasicML/examples/train_lenet_adamw.py

Rewrite of ``train_lenet_mnist.py`` with the Muon comparison stripped back
out (AdamW alone won that comparison -- 98.40% best test accuracy vs.
Muon+AdamW's divergence to 73.00%, see TODO-0026) and every hyperparameter
pulled out into ``lenet_config.LeNetConfig`` instead of module-level
constants. That split exists so ``tune_lenet_adamw.py`` can search over
many ``LeNetConfig``s and call straight into this file's ``build_model``/
``train`` -- the model and training loop are defined exactly once.
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
from basicml.nn.pool       import MaxPool2D
from basicml.nn.flatten    import Flatten
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Softmax
from basicml.nn.loss       import CrossEntropyLoss
from basicml.optim.adamw   import AdamW
from basicml.datasets      import iter_minibatches
from basicml.metrics       import Accuracy
from examples.lenet_config import LeNetConfig, DEFAULT_CONFIG, load_config

np.set_printoptions(suppress=True, precision=4)

SHOW_PLOT = True


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


def load_mnist_split(cfg: LeNetConfig) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch real MNIST via scikit-learn and split into disjoint train/test sets.

    Args:
        cfg: Run configuration (uses ``n_train``, ``n_test``, ``seed``,
            ``image_size``).

    Returns:
        Tuple ``(X_train, y_train_int, y_train_onehot, X_test, y_test_int,
        y_test_onehot)``.
    """
    print("Fetching MNIST (cached after first download)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    rng     = np.random.RandomState(cfg.seed)
    indices = rng.permutation(len(mnist.data))
    train_idx = indices[:cfg.n_train]
    test_idx  = indices[cfg.n_train:cfg.n_train + cfg.n_test]

    def prepare(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = mnist.data[idx].astype(np.float64) / 255.0
        X = X.reshape(-1, 1, cfg.image_size, cfg.image_size)
        y_int    = mnist.target[idx].astype(np.int64)
        y_onehot = np.eye(10)[y_int]
        return X, y_int, y_onehot

    X_train, y_train_int, y_train_onehot = prepare(train_idx)
    X_test, y_test_int, y_test_onehot    = prepare(test_idx)
    return X_train, y_train_int, y_train_onehot, X_test, y_test_int, y_test_onehot


def build_model(cfg: LeNetConfig) -> dict[str, Module]:
    """Build the LeNet-shaped classifier as a dict of named ``basicml.nn`` layers.

    Args:
        cfg: Run configuration (uses ``seed``, ``image_size``, ``conv1_ch``,
            ``conv2_ch``, ``kernel1``, ``kernel2``, ``fc1_units``,
            ``fc2_units``).

    Returns:
        Dict mapping layer name to layer instance, in forward order.
    """
    np.random.seed(cfg.seed)
    conv1_out_size = cfg.image_size - cfg.kernel1 + 1
    pool1_out_size = conv1_out_size // 2
    conv2_out_size = pool1_out_size - cfg.kernel2 + 1
    pool2_out_size = conv2_out_size // 2
    flat_features  = cfg.conv2_ch * pool2_out_size ** 2

    return {
        "conv1"  : Conv2D(1, cfg.conv1_ch, kernel_size=cfg.kernel1, stride=1, padding=0, init_type="he"),
        "relu1"  : ReLU(),
        "pool1"  : MaxPool2D(2),
        "conv2"  : Conv2D(cfg.conv1_ch, cfg.conv2_ch, kernel_size=cfg.kernel2, stride=1, padding=0, init_type="he"),
        "relu2"  : ReLU(),
        "pool2"  : MaxPool2D(2),
        "flatten": Flatten(),
        "fc1"    : Linear(flat_features, cfg.fc1_units, init_type="he"),
        "relu3"  : ReLU(),
        "fc2"    : Linear(cfg.fc1_units, cfg.fc2_units, init_type="he"),
        "relu4"  : ReLU(),
        "fc3"    : Linear(cfg.fc2_units, 10, init_type="xavier"),
        "softmax": Softmax(axis=-1),
    }


_FORWARD_ORDER = ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "flatten",
                  "fc1", "relu3", "fc2", "relu4", "fc3", "softmax")


def forward(layers: dict[str, Module], X: np.ndarray) -> np.ndarray:
    """Run ``X`` through every layer in order.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        X: Input batch, shape ``(N, 1, image_size, image_size)``.

    Returns:
        Softmax class probabilities, shape ``(N, 10)``.
    """
    out = X
    for name in _FORWARD_ORDER:
        out = layers[name](out)
    return out


def backward(layers: dict[str, Module], grad: np.ndarray) -> None:
    """Manually chain ``backward()`` through every layer, in reverse order.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        grad: Upstream gradient from the loss, shape matching the model's
            output.
    """
    for name in reversed(_FORWARD_ORDER):
        grad = layers[name].backward(grad)


def parameters(layers: dict[str, Module]) -> list:
    """Collect every trainable parameter across the model's layers.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        Flat list of ``Tensor`` parameters, in layer order.
    """
    params = []
    for name in ("conv1", "conv2", "fc1", "fc2", "fc3"):
        params.extend(layers[name].parameters())
    return params


def evaluate(layers: dict[str, Module], criterion: CrossEntropyLoss,
            X: np.ndarray, y_onehot: np.ndarray, batch_size: int = 256) -> tuple[float, float]:
    """Compute mean loss and accuracy over a dataset.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        criterion: Loss used to score each batch.
        X: Images to evaluate, shape ``(N, 1, image_size, image_size)``.
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


def train(cfg: LeNetConfig, layers: dict[str, Module], X_train: np.ndarray, y_train_onehot: np.ndarray,
          X_test: np.ndarray, y_test_onehot: np.ndarray, verbose: bool = True) -> list[EpochLog]:
    """Train the model with AdamW, tracking train/test metrics per epoch.

    Args:
        cfg: Run configuration (uses ``epochs``, ``batch_size``, ``lr``,
            ``weight_decay``, ``seed``).
        layers: Model layers, as returned by :func:`build_model`.
        X_train: Training images, shape ``(n_train, 1, image_size, image_size)``.
        y_train_onehot: One-hot training labels.
        X_test: Held-out test images, shape ``(n_test, 1, image_size, image_size)``.
        y_test_onehot: One-hot test labels.
        verbose: Whether to print a progress line per epoch -- silenced by
            ``tune_lenet_adamw.py`` so a search over many configs doesn't
            flood stdout.

    Returns:
        One :class:`EpochLog` per epoch, in training order.
    """
    criterion = CrossEntropyLoss()
    optimizer = AdamW(parameters(layers), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logs      = []

    for epoch in range(1, cfg.epochs + 1):
        train_accuracy   = Accuracy(num_classes=10)
        train_loss_total = 0.0
        n_batches        = 0
        for xb, yb in iter_minibatches(X_train, y_train_onehot, cfg.batch_size, random_state=cfg.seed + epoch):
            probs = forward(layers, xb)
            cost  = criterion(probs, yb)

            backward(layers, criterion.backward())
            optimizer.step()
            optimizer.zero_grad()

            train_accuracy.update(probs, yb)
            train_loss_total += cost
            n_batches        += 1

        train_loss = train_loss_total / n_batches
        train_acc  = float(train_accuracy.compute())
        test_loss, test_acc = evaluate(layers, criterion, X_test, y_test_onehot)
        logs.append(EpochLog(epoch, train_loss, train_acc, test_loss, test_acc))
        if verbose:
            print(f"epoch {epoch:>2}/{cfg.epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                 f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    return logs


def plot_curves(logs: list[EpochLog]) -> None:
    """Plot train/test loss and accuracy curves across epochs.

    Args:
        logs: Per-epoch metrics, as returned by :func:`train`.
    """
    epochs = [log.epoch for log in logs]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - LeNet on MNIST (AdamW)")

    ax_loss.plot(epochs, [log.train_loss for log in logs], marker="o", label="train")
    ax_loss.plot(epochs, [log.test_loss for log in logs], marker="o", label="test")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [log.train_acc for log in logs], marker="o", label="train")
    ax_acc.plot(epochs, [log.test_acc for log in logs], marker="o", label="test")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def run(cfg: LeNetConfig, show_plot: bool = SHOW_PLOT) -> list[EpochLog]:
    """Fetch MNIST, train the LeNet-shaped model with AdamW, and report test accuracy.

    Args:
        cfg: Run configuration to train with.
        show_plot: Whether to display the loss/accuracy plot at the end.

    Returns:
        One :class:`EpochLog` per epoch, in training order -- so callers
        like ``tune_lenet_adamw.py`` can read off the final test accuracy
        without re-parsing stdout.
    """
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = load_mnist_split(cfg)
    layers = build_model(cfg)
    logs   = train(cfg, layers, X_train, y_train_onehot, X_test, y_test_onehot)

    print(f"\nFinal held-out test accuracy: {logs[-1].test_acc:.4f} "
         f"on {cfg.n_test} unseen images (trained on {cfg.n_train})")
    if show_plot:
        plot_curves(logs)
    return logs


def main() -> None:
    """Entry point: train once, with ``DEFAULT_CONFIG`` or a ``--config`` file.

    ``tune_lenet_adamw.py`` only searches and writes its winning
    ``LeNetConfig`` to a ``.cfg`` file -- it never trains at full budget
    itself. Passing that file here is how the two steps connect:

        python BasicML/examples/tune_lenet_adamw.py            # writes lenet_best.cfg
        python BasicML/examples/train_lenet_adamw.py --config BasicML/examples/lenet_best.cfg
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a .cfg file written by tune_lenet_adamw.py "
                             "(defaults to lenet_config.DEFAULT_CONFIG if omitted).")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    run(cfg)


if __name__ == "__main__":
    main()
