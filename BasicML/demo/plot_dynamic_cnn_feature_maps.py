# AI generated (refactored/authored with Claude Code)
"""Animated feature-map evolution of a small CNN training on real MNIST.

Run directly: python BasicML/demo/plot_dynamic_cnn_feature_maps.py

Trains a ``Conv2D -> ReLU -> MaxPool2D`` x2 -> ``Flatten -> Linear ->
Softmax`` classifier (all ``basicml.nn``, manual backward -- no autograd) on
a subset of real handwritten digits from MNIST, fetched once via
``sklearn.datasets.fetch_openml`` and cached locally by scikit-learn. One
fixed input digit is re-run through the network after every epoch, and its
two convolutional layers' feature maps are recorded; the recorded snapshots
are then played back as an animation so a beginner can watch the filters go
from meaningless random-init noise to edge/stroke detectors as training
progresses -- the flip side of the *static*, untrained
``plot_feature_maps_receptive_field.py`` demo in this same folder.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from sklearn.datasets import fetch_openml

from basicml.nn.module     import Module
from basicml.nn.conv       import Conv2D
from basicml.nn.pool       import MaxPool2D
from basicml.nn.flatten    import Flatten
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Softmax
from basicml.nn.loss       import CrossEntropyLoss
from basicml.optim.adam    import Adam
from basicml.datasets      import iter_minibatches
from basicml.metrics       import Accuracy

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ----------------------------------------------------------------
SEED          = 0
N_TRAIN       = 3000
IMAGE_SIZE    = 28
TARGET_DIGIT  = 8   # digit tracked/animated across training epochs
CONV1_CH      = 8
CONV2_CH      = 16
EPOCHS        = 15
BATCH_SIZE    = 64
LEARN_RATE    = 0.01
MAX_CH_SHOWN  = 8
FRAME_INTERVAL = 500
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """One epoch's worth of state for the animation.

    Attributes:
        epoch: Epoch number (1-indexed).
        train_loss: Mean cross-entropy loss over that epoch.
        train_acc: Training accuracy over that epoch.
        conv1_maps: Fixed sample's conv1 feature maps, shape (CONV1_CH, 28, 28).
        conv2_maps: Fixed sample's conv2 feature maps, shape (CONV2_CH, 14, 14).
        probs: Predicted class probabilities for the fixed sample, shape (10,).
    """
    epoch     : int
    train_loss: float
    train_acc : float
    conv1_maps: np.ndarray
    conv2_maps: np.ndarray
    probs     : np.ndarray


def load_mnist_subset(n_train: int, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch real MNIST via scikit-learn and take a small labeled subset.

    Downloads once (``fetch_openml`` caches to disk afterward), normalizes
    pixels to ``[0, 1]``, and one-hot encodes the digit labels.

    Args:
        n_train: Number of images to keep for training.
        random_state: Seed for the subsampling permutation.

    Returns:
        Tuple ``(X, y_int, y_onehot)`` where ``X`` has shape
        ``(n_train, 1, 28, 28)``, ``y_int`` has shape ``(n_train,)``, and
        ``y_onehot`` has shape ``(n_train, 10)``.
    """
    print("Fetching MNIST (cached after first download)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    rng     = np.random.RandomState(random_state)
    indices = rng.permutation(len(mnist.data))[:n_train]

    X = mnist.data[indices].astype(np.float64) / 255.0
    X = X.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    y_int    = mnist.target[indices].astype(np.int64)
    y_onehot = np.eye(10)[y_int]
    return X, y_int, y_onehot


def pick_sample(X: np.ndarray, y_int: np.ndarray, digit: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Pick the first training image whose label matches ``digit``.

    Args:
        X: Training images, shape ``(N, 1, 28, 28)``.
        y_int: Integer training labels, shape ``(N,)``.
        digit: The digit class (``0``-``9``) to track across training.

    Returns:
        Tuple ``(sample_x, sample_image, true_label)`` -- a batch-of-one
        input, the same image squeezed to ``(28, 28)`` for plotting, and its
        integer label (always ``digit``).

    Raises:
        ValueError: If ``digit`` does not appear anywhere in ``y_int``.
    """
    matches = np.where(y_int == digit)[0]
    if len(matches) == 0:
        raise ValueError(f"no training sample with label {digit} in this subset "
                         f"(try a larger N_TRAIN or a different TARGET_DIGIT)")
    index = int(matches[0])
    return X[index:index + 1], X[index, 0], digit


def build_model(seed: int) -> dict[str, Module]:
    """Build the CNN classifier as a plain dict of named ``basicml.nn`` layers.

    Kept as a flat dict of layers -- rather than a ``models.CNNModel``
    subclass -- so this script can grab the two conv layers' outputs
    directly for the feature-map animation.

    Args:
        seed: Seed for the ``Conv2D``/``Linear`` weight initialization.

    Returns:
        Dict mapping layer name to layer instance, in forward order.
    """
    np.random.seed(seed)
    return {
        "conv1"  : Conv2D(1, CONV1_CH, kernel_size=5, stride=1, padding=2, init_type="he"),
        "relu1"  : ReLU(),
        "pool1"  : MaxPool2D(2),
        "conv2"  : Conv2D(CONV1_CH, CONV2_CH, kernel_size=5, stride=1, padding=2, init_type="he"),
        "relu2"  : ReLU(),
        "pool2"  : MaxPool2D(2),
        "flatten": Flatten(),
        "linear" : Linear(CONV2_CH * (IMAGE_SIZE // 4) ** 2, 10, init_type="xavier"),
        "softmax": Softmax(axis=-1),
    }


def forward(layers: dict[str, Module], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``X`` through every layer in order, keeping both conv outputs.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        X: Input batch, shape ``(N, 1, 28, 28)``.

    Returns:
        Tuple ``(probs, conv1_out, conv2_out)`` -- final softmax
        probabilities plus each conv layer's post-ReLU feature maps.
    """
    out        = layers["conv1"](X)
    out        = layers["relu1"](out)
    conv1_out  = out
    out        = layers["pool1"](out)
    out        = layers["conv2"](out)
    out        = layers["relu2"](out)
    conv2_out  = out
    out        = layers["pool2"](out)
    out        = layers["flatten"](out)
    out        = layers["linear"](out)
    probs      = layers["softmax"](out)
    return probs, conv1_out, conv2_out


def backward(layers: dict[str, Module], grad: np.ndarray) -> None:
    """Manually chain ``backward()`` through every layer, in reverse order.

    Mirrors ``nn.models.CNNModel.backward`` -- each layer's own cached
    ``forward()`` state drives its local gradient; this function is only
    responsible for the order.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        grad: Upstream gradient from the loss, shape matching the model's
            output.
    """
    grad = layers["softmax"].backward(grad)
    grad = layers["linear"].backward(grad)
    grad = layers["flatten"].backward(grad)
    grad = layers["pool2"].backward(grad)
    grad = layers["relu2"].backward(grad)
    grad = layers["conv2"].backward(grad)
    grad = layers["pool1"].backward(grad)
    grad = layers["relu1"].backward(grad)
    layers["conv1"].backward(grad)


def parameters(layers: dict[str, Module]) -> list:
    """Collect every trainable parameter across the model's layers.

    Args:
        layers: Model layers, as returned by :func:`build_model`.

    Returns:
        Flat list of ``Tensor`` parameters, in layer order.
    """
    params = []
    for name in ("conv1", "conv2", "linear"):
        params.extend(layers[name].parameters())
    return params


def train_and_record(layers: dict[str, Module], X: np.ndarray, y_onehot: np.ndarray,
                     sample_x: np.ndarray) -> list[Snapshot]:
    """Train the model while recording one snapshot per epoch.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        X: Training images, shape ``(N, 1, 28, 28)``.
        y_onehot: One-hot training labels, shape ``(N, 10)``.
        sample_x: The single fixed image tracked across epochs, shape
            ``(1, 1, 28, 28)``.

    Returns:
        One :class:`Snapshot` per epoch, in training order.
    """
    criterion = CrossEntropyLoss()
    optimizer = Adam(parameters(layers), lr=LEARN_RATE)
    snapshots = []

    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        accuracy    = Accuracy(num_classes=10)
        epoch_loss  = 0.0
        n_batches   = 0
        for xb, yb in iter_minibatches(X, y_onehot, BATCH_SIZE, random_state=SEED + epoch):
            probs, _, _ = forward(layers, xb)
            loss        = criterion(probs, yb)
            backward(layers, criterion.backward())
            optimizer.step()
            optimizer.zero_grad()

            accuracy.update(probs, yb)
            epoch_loss += loss
            n_batches  += 1

        train_loss = epoch_loss / n_batches
        train_acc  = accuracy.compute()
        sample_probs, conv1_out, conv2_out = forward(layers, sample_x)
        snapshots.append(Snapshot(
            epoch      = epoch,
            train_loss = train_loss,
            train_acc  = float(train_acc),
            conv1_maps = conv1_out[0].copy(),
            conv2_maps = conv2_out[0].copy(),
            probs      = sample_probs[0].copy(),
        ))
        print(f"epoch {epoch:>2}/{EPOCHS}  loss={train_loss:.4f}  acc={train_acc:.4f}")

    return snapshots


def animate(sample_image: np.ndarray, true_label: int, snapshots: list[Snapshot]) -> FuncAnimation:
    """Play the recorded per-epoch feature maps back as an animation.

    Args:
        sample_image: The fixed tracked digit, shape ``(28, 28)``.
        true_label: That digit's ground-truth class, ``0``-``9``.
        snapshots: Per-epoch state, as returned by :func:`train_and_record`.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation` handle (kept alive
        so the animation is not garbage-collected).
    """
    n_cols = MAX_CH_SHOWN
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 1.5, 3 * 1.7))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - CNN Feature Maps During Training")

    axes[0, 0].imshow(sample_image, cmap="gray")
    axes[0, 0].set_ylabel(f"input\ndigit={true_label}", fontsize=8)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    for ax in axes[0, 1:]:
        ax.axis("off")

    conv1_images = []
    for col in range(n_cols):
        ax = axes[1, col]
        im = ax.imshow(np.zeros((28, 28)), cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        conv1_images.append(im)
    axes[1, 0].set_ylabel(f"conv1\n{CONV1_CH}ch", fontsize=8)

    conv2_images = []
    for col in range(n_cols):
        ax = axes[2, col]
        im = ax.imshow(np.zeros((14, 14)), cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        conv2_images.append(im)
    axes[2, 0].set_ylabel(f"conv2\nfirst {n_cols}/{CONV2_CH}ch", fontsize=8)

    def update(frame: int):
        """Redraw every feature map and the title for epoch ``frame``."""
        snap = snapshots[frame]
        for col in range(n_cols):
            fmap = snap.conv1_maps[col]
            im   = conv1_images[col]
            im.set_data(fmap)
            im.set_clim(fmap.min(), max(fmap.max(), fmap.min() + 1e-8))

            fmap2 = snap.conv2_maps[col]
            im2   = conv2_images[col]
            im2.set_data(fmap2)
            im2.set_clim(fmap2.min(), max(fmap2.max(), fmap2.min() + 1e-8))

        pred = int(np.argmax(snap.probs))
        fig.suptitle(f"Epoch {snap.epoch}/{EPOCHS}  |  loss={snap.train_loss:.4f}  "
                    f"acc={snap.train_acc:.4f}  |  predicted={pred} "
                    f"({snap.probs[pred]:.2f})  true={true_label}")
        return []

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         interval=FRAME_INTERVAL, blit=False, repeat=True)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()
    return anim


def main() -> None:
    """Fetch MNIST, train while recording feature maps, then animate them."""
    X, y_int, y_onehot = load_mnist_subset(N_TRAIN, SEED)
    sample_x, sample_image, true_label = pick_sample(X, y_int, TARGET_DIGIT)

    layers    = build_model(SEED)
    snapshots = train_and_record(layers, X, y_onehot, sample_x)
    animate(sample_image, true_label, snapshots)


if __name__ == "__main__":
    main()
