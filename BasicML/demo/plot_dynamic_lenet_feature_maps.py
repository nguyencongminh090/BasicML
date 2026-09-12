# AI generated (refactored/authored with Claude Code)
"""Animated feature-map evolution of a LeNet-ish CNN training on real MNIST.

Run directly: python BasicML/demo/plot_dynamic_lenet_feature_maps.py

A "modernized LeNet" -- keeps the original LeNet-5 (LeCun et al., 1998)
layer *shapes* (6 then 16 conv channels, 5x5 kernels, 120- then 84-unit
fully-connected layers before the 10-way output) but swaps its original
Tanh/AvgPool for ReLU/MaxPool2D, and skips the classic 28x28 -> 32x32
zero-padding (so the two 5x5 "valid" convolutions end at 24x24 -> 4x4
instead of 28x28 -> 5x5). All layers are ``basicml.nn`` with manual
backward -- no autograd.

Same recording/animation idea as the sibling
``plot_dynamic_cnn_feature_maps.py`` demo (same folder): one fixed digit is
re-run through the network after every training epoch, its conv1/conv2
feature maps are recorded, and the recorded snapshots are played back as an
animation so a beginner can watch a LeNet-shaped network's filters sharpen
during training -- and compare that against the other, differently-shaped
small CNN in the sibling demo.
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
from basicml.nn.pool       import MaxPool2D, AvgPool2D
from basicml.nn.flatten    import Flatten
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Softmax
from basicml.nn.loss       import CrossEntropyLoss
from basicml.optim.adam    import Adam
from basicml.datasets      import iter_minibatches
from basicml.metrics       import Accuracy

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ----------------------------------------------------------------
SEED           = 0
N_TRAIN        = 3000
IMAGE_SIZE     = 28
TARGET_DIGIT   = 7     # digit tracked/animated across training epochs
CONV1_CH       = 6     # LeNet-5's original C1 channel count
CONV2_CH       = 16    # LeNet-5's original C3 channel count
FC1_UNITS      = 120   # LeNet-5's original F5
FC2_UNITS      = 84    # LeNet-5's original F6
EPOCHS         = 15
BATCH_SIZE     = 64
LEARN_RATE     = 0.01
MAX_CH_SHOWN   = 8
FRAME_INTERVAL = 500
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """One epoch's worth of state for the animation.

    Attributes:
        epoch: Epoch number (1-indexed).
        train_loss: Mean cross-entropy loss over that epoch.
        train_acc: Training accuracy over that epoch.
        conv1_maps: Fixed sample's conv1 feature maps, shape (CONV1_CH, 24, 24).
        conv2_maps: Fixed sample's conv2 feature maps, shape (CONV2_CH, 8, 8).
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
    """Build the LeNet-ish classifier as a plain dict of named ``basicml.nn`` layers.

    Kept as a flat dict of layers -- rather than a ``models.CNNModel``
    subclass -- so this script can grab the two conv layers' outputs
    directly for the feature-map animation.

    Args:
        seed: Seed for the ``Conv2D``/``Linear`` weight initialization.

    Returns:
        Dict mapping layer name to layer instance, in forward order.
    """
    np.random.seed(seed)
    conv1_out_size = IMAGE_SIZE - 5 + 1               # 28 -> 24, valid 5x5 conv
    pool1_out_size = conv1_out_size // 2              # 24 -> 12
    conv2_out_size = pool1_out_size - 5 + 1           # 12 -> 8
    pool2_out_size = conv2_out_size // 2              # 8 -> 4
    flat_features  = CONV2_CH * pool2_out_size ** 2   # 16 * 4 * 4 = 256

    return {
        "conv1"  : Conv2D(1, CONV1_CH, kernel_size=5, stride=1, padding=0, init_type="he"),
        "relu1"  : ReLU(),
        "pool1"  : AvgPool2D(2),
        "conv2"  : Conv2D(CONV1_CH, CONV2_CH, kernel_size=5, stride=1, padding=0, init_type="he"),
        "relu2"  : ReLU(),
        "pool2"  : AvgPool2D(2),
        "flatten": Flatten(),
        "fc1"    : Linear(flat_features, FC1_UNITS, init_type="he"),
        "relu3"  : ReLU(),
        "fc2"    : Linear(FC1_UNITS, FC2_UNITS, init_type="he"),
        "relu4"  : ReLU(),
        "fc3"    : Linear(FC2_UNITS, 10, init_type="xavier"),
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
    out       = layers["conv1"](X)
    out       = layers["relu1"](out)
    conv1_out = out
    out       = layers["pool1"](out)
    out       = layers["conv2"](out)
    out       = layers["relu2"](out)
    conv2_out = out
    out       = layers["pool2"](out)
    out       = layers["flatten"](out)
    out       = layers["fc1"](out)
    out       = layers["relu3"](out)
    out       = layers["fc2"](out)
    out       = layers["relu4"](out)
    out       = layers["fc3"](out)
    probs     = layers["softmax"](out)
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
    grad = layers["fc3"].backward(grad)
    grad = layers["relu4"].backward(grad)
    grad = layers["fc2"].backward(grad)
    grad = layers["relu3"].backward(grad)
    grad = layers["fc1"].backward(grad)
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
    for name in ("conv1", "conv2", "fc1", "fc2", "fc3"):
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
        accuracy   = Accuracy(num_classes=10)
        epoch_loss = 0.0
        n_batches  = 0
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
        fig.canvas.manager.set_window_title("BasicML - LeNet Feature Maps During Training")

    axes[0, 0].imshow(sample_image, cmap="gray")
    axes[0, 0].set_ylabel(f"input\ndigit={true_label}", fontsize=8)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    for ax in axes[0, 1:]:
        ax.axis("off")

    conv1_shown  = min(CONV1_CH, n_cols)
    conv1_images = []
    for col in range(n_cols):
        ax = axes[1, col]
        if col < conv1_shown:
            im = ax.imshow(np.zeros((24, 24)), cmap="viridis")
            conv1_images.append(im)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].set_ylabel(f"conv1\n{CONV1_CH}ch", fontsize=8)

    conv2_shown  = min(CONV2_CH, n_cols)
    conv2_images = []
    for col in range(n_cols):
        ax = axes[2, col]
        if col < conv2_shown:
            im = ax.imshow(np.zeros((8, 8)), cmap="viridis")
            conv2_images.append(im)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])
    axes[2, 0].set_ylabel(f"conv2\nfirst {conv2_shown}/{CONV2_CH}ch", fontsize=8)

    def update(frame: int):
        """Redraw every feature map and the title for epoch ``frame``."""
        snap = snapshots[frame]
        for col in range(conv1_shown):
            fmap = snap.conv1_maps[col]
            im   = conv1_images[col]
            im.set_data(fmap)
            im.set_clim(fmap.min(), max(fmap.max(), fmap.min() + 1e-8))

        for col in range(conv2_shown):
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
    """Fetch MNIST, train the LeNet-ish model while recording feature maps, then animate them."""
    X, y_int, y_onehot = load_mnist_subset(N_TRAIN, SEED)
    sample_x, sample_image, true_label = pick_sample(X, y_int, TARGET_DIGIT)

    layers    = build_model(SEED)
    snapshots = train_and_record(layers, X, y_onehot, sample_x)
    animate(sample_image, true_label, snapshots)


if __name__ == "__main__":
    main()
