# AI generated (refactored/authored with Claude Code)
"""Animated feature-map evolution of an AlexNet-shaped CNN training on real MNIST.

Run directly: python BasicML/demo/plot_dynamic_alexnet_feature_maps.py

AlexNet (Krizhevsky et al., 2012) was built for 227x227x3 ImageNet images:
5 conv layers (96 -> 256 -> 384 -> 384 -> 256 channels, kernels up to 11x11
stride 4), only 3 max-pools (after conv1, conv2, and conv5 -- conv3/4/5 run
back-to-back with no pooling between them), 2 big fully-connected layers
with Dropout, and ReLU throughout (AlexNet is largely what popularized ReLU
over Tanh/Sigmoid for CNNs). Applied literally to 28x28x1 MNIST the first
conv layer alone would collapse the image to a few pixels before conv2 even
runs, so this demo keeps AlexNet's *defining shape* -- 5 conv layers whose
channel count grows then plateaus, pooling only after conv1/conv2/conv5,
2 Dropout-regularized FC layers, ReLU everywhere -- but with 3x3/stride-1
convolutions and channel counts sized for MNIST instead of ImageNet. Local
Response Normalization (the original's other novelty) is skipped: it has no
``basicml.nn`` implementation and was later shown to add little over what
Dropout/BatchNorm already provide, so nothing in this repo's architecture
needs it added.

Same recording/animation idea as the sibling ``plot_dynamic_cnn_feature_maps.py``
and ``plot_dynamic_lenet_feature_maps.py`` demos (same folder): one fixed
digit is re-run through the network (in eval mode, so Dropout is off) after
every training epoch, three representative conv layers' feature maps are
recorded, and the recorded snapshots are played back as an animation.
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
from basicml.nn.dropout    import Dropout
from basicml.nn.activation import ReLU, Softmax
from basicml.nn.loss       import CrossEntropyLoss
from basicml.optim.adam    import Adam
from basicml.datasets      import iter_minibatches
from basicml.metrics       import Accuracy

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ----------------------------------------------------------------
SEED           = 0
# Smaller than the sibling demos' N_TRAIN=3000/EPOCHS=15: this model has 5
# conv layers plus two 512/256-wide FC layers, so pure-numpy tap-loop
# training is noticeably slower (~4-5 min end to end at these settings vs.
# ~1 min for the LeNet-shaped demo).
N_TRAIN        = 1200
IMAGE_SIZE     = 28
TARGET_DIGIT   = 8      # digit tracked/animated across training epochs

# Scaled-down AlexNet channel pattern: grows then plateaus, like the
# original's 96 -> 256 -> 384 -> 384 -> 256.
CONV1_CH       = 32
CONV2_CH       = 64
CONV3_CH       = 96
CONV4_CH       = 96
CONV5_CH       = 64
FC1_UNITS      = 512
FC2_UNITS      = 256
DROPOUT_P      = 0.3    # lighter than AlexNet's original 0.5 -- this training set is tiny by comparison

EPOCHS         = 10
BATCH_SIZE     = 64
LEARN_RATE     = 0.002
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
        conv1_maps: Fixed sample's conv1 feature maps, shape (CONV1_CH, 28, 28).
        conv3_maps: Fixed sample's conv3 feature maps, shape (CONV3_CH, 7, 7).
        conv5_maps: Fixed sample's conv5 feature maps, shape (CONV5_CH, 7, 7).
        probs: Predicted class probabilities for the fixed sample, shape (10,).
    """
    epoch     : int
    train_loss: float
    train_acc : float
    conv1_maps: np.ndarray
    conv3_maps: np.ndarray
    conv5_maps: np.ndarray
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
    """Build the AlexNet-shaped classifier as a plain dict of named ``basicml.nn`` layers.

    Kept as a flat dict of layers -- rather than a ``models.CNNModel``
    subclass -- so this script can grab intermediate conv layers' outputs
    directly for the feature-map animation.

    Args:
        seed: Seed for the ``Conv2D``/``Linear`` weight initialization.

    Returns:
        Dict mapping layer name to layer instance, in forward order.
    """
    np.random.seed(seed)
    pool1_out = IMAGE_SIZE // 2                  # 28 -> 14
    pool2_out = pool1_out // 2                    # 14 -> 7
    pool3_out = (pool2_out - 2) // 2 + 1          # 7 -> 3 (conv3/4/5 keep spatial size)
    flat_features = CONV5_CH * pool3_out ** 2     # 64 * 3 * 3 = 576

    return {
        "conv1"  : Conv2D(1, CONV1_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "relu1"  : ReLU(),
        "pool1"  : MaxPool2D(2),
        "conv2"  : Conv2D(CONV1_CH, CONV2_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "relu2"  : ReLU(),
        "pool2"  : MaxPool2D(2),
        "conv3"  : Conv2D(CONV2_CH, CONV3_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "relu3"  : ReLU(),
        "conv4"  : Conv2D(CONV3_CH, CONV4_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "relu4"  : ReLU(),
        "conv5"  : Conv2D(CONV4_CH, CONV5_CH, kernel_size=3, stride=1, padding=1, init_type="he"),
        "relu5"  : ReLU(),
        "pool3"  : MaxPool2D(2),
        "flatten": Flatten(),
        "fc1"    : Linear(flat_features, FC1_UNITS, init_type="he"),
        "relu6"  : ReLU(),
        "drop1"  : Dropout(DROPOUT_P),
        "fc2"    : Linear(FC1_UNITS, FC2_UNITS, init_type="he"),
        "relu7"  : ReLU(),
        "drop2"  : Dropout(DROPOUT_P),
        "fc3"    : Linear(FC2_UNITS, 10, init_type="xavier"),
        "softmax": Softmax(axis=-1),
    }


def set_training(layers: dict[str, Module], mode: bool) -> None:
    """Toggle every layer's ``training`` flag (only ``Dropout`` reacts to it).

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        mode: ``True`` for training (Dropout active), ``False`` for eval
            (Dropout is a no-op, giving deterministic predictions).
    """
    for layer in layers.values():
        layer.train(mode)


def forward(layers: dict[str, Module], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``X`` through every layer in order, keeping three conv outputs.

    Args:
        layers: Model layers, as returned by :func:`build_model`.
        X: Input batch, shape ``(N, 1, 28, 28)``.

    Returns:
        Tuple ``(probs, conv1_out, conv3_out, conv5_out)`` -- final softmax
        probabilities plus three representative conv layers' post-ReLU
        feature maps (early/mid/late in the stack).
    """
    out       = layers["conv1"](X)
    out       = layers["relu1"](out)
    conv1_out = out
    out       = layers["pool1"](out)
    out       = layers["conv2"](out)
    out       = layers["relu2"](out)
    out       = layers["pool2"](out)
    out       = layers["conv3"](out)
    out       = layers["relu3"](out)
    conv3_out = out
    out       = layers["conv4"](out)
    out       = layers["relu4"](out)
    out       = layers["conv5"](out)
    out       = layers["relu5"](out)
    conv5_out = out
    out       = layers["pool3"](out)
    out       = layers["flatten"](out)
    out       = layers["fc1"](out)
    out       = layers["relu6"](out)
    out       = layers["drop1"](out)
    out       = layers["fc2"](out)
    out       = layers["relu7"](out)
    out       = layers["drop2"](out)
    out       = layers["fc3"](out)
    probs     = layers["softmax"](out)
    return probs, conv1_out, conv3_out, conv5_out


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
    grad = layers["drop2"].backward(grad)
    grad = layers["relu7"].backward(grad)
    grad = layers["fc2"].backward(grad)
    grad = layers["drop1"].backward(grad)
    grad = layers["relu6"].backward(grad)
    grad = layers["fc1"].backward(grad)
    grad = layers["flatten"].backward(grad)
    grad = layers["pool3"].backward(grad)
    grad = layers["relu5"].backward(grad)
    grad = layers["conv5"].backward(grad)
    grad = layers["relu4"].backward(grad)
    grad = layers["conv4"].backward(grad)
    grad = layers["relu3"].backward(grad)
    grad = layers["conv3"].backward(grad)
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
    for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "fc3"):
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
        set_training(layers, True)
        accuracy   = Accuracy(num_classes=10)
        epoch_loss = 0.0
        n_batches  = 0
        for xb, yb in iter_minibatches(X, y_onehot, BATCH_SIZE, random_state=SEED + epoch):
            probs, _, _, _ = forward(layers, xb)
            loss           = criterion(probs, yb)
            backward(layers, criterion.backward())
            optimizer.step()
            optimizer.zero_grad()

            accuracy.update(probs, yb)
            epoch_loss += loss
            n_batches  += 1

        train_loss = epoch_loss / n_batches
        train_acc  = accuracy.compute()

        set_training(layers, False)
        sample_probs, conv1_out, conv3_out, conv5_out = forward(layers, sample_x)
        snapshots.append(Snapshot(
            epoch      = epoch,
            train_loss = train_loss,
            train_acc  = float(train_acc),
            conv1_maps = conv1_out[0].copy(),
            conv3_maps = conv3_out[0].copy(),
            conv5_maps = conv5_out[0].copy(),
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
    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 1.5, 4 * 1.6))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - AlexNet-shaped Feature Maps During Training")

    axes[0, 0].imshow(sample_image, cmap="gray")
    axes[0, 0].set_ylabel(f"input\ndigit={true_label}", fontsize=8)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    for ax in axes[0, 1:]:
        ax.axis("off")

    def make_row(row: int, channels: int, spatial: int, label: str) -> list:
        shown = min(channels, n_cols)
        images = []
        for col in range(n_cols):
            ax = axes[row, col]
            if col < shown:
                images.append(ax.imshow(np.zeros((spatial, spatial)), cmap="viridis"))
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{label}\nfirst {shown}/{channels}ch", fontsize=8)
        return images

    conv1_images = make_row(1, CONV1_CH, 28, "conv1")
    conv3_images = make_row(2, CONV3_CH, 7, "conv3")
    conv5_images = make_row(3, CONV5_CH, 7, "conv5")

    def refresh(images: list, maps: np.ndarray) -> None:
        for col, im in enumerate(images):
            fmap = maps[col]
            im.set_data(fmap)
            im.set_clim(fmap.min(), max(fmap.max(), fmap.min() + 1e-8))

    def update(frame: int):
        """Redraw every feature map and the title for epoch ``frame``."""
        snap = snapshots[frame]
        refresh(conv1_images, snap.conv1_maps)
        refresh(conv3_images, snap.conv3_maps)
        refresh(conv5_images, snap.conv5_maps)

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
    """Fetch MNIST, train the AlexNet-shaped model while recording feature maps, then animate them."""
    X, y_int, y_onehot = load_mnist_subset(N_TRAIN, SEED)
    sample_x, sample_image, true_label = pick_sample(X, y_int, TARGET_DIGIT)

    layers    = build_model(SEED)
    snapshots = train_and_record(layers, X, y_onehot, sample_x)
    animate(sample_image, true_label, snapshots)


if __name__ == "__main__":
    main()
