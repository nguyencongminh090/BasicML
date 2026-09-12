# AI generated (refactored/authored with Claude Code)
"""Feature maps and receptive field growth in a small CNN stack.

Run directly: python BasicML/demo/plot_feature_maps_receptive_field.py

Builds a small ``Conv2D -> ReLU -> MaxPool2D`` stack (from ``basicml.nn``)
over one synthetic image with clear geometric edges, then answers, visually,
three beginner questions about CNNs:

1. Why does the spatial size shrink layer by layer?
   -> a grid of the feature maps produced by each layer.
2. Why does receptive field grow so much faster with stride/pooling than
   with depth alone?
   -> ``RF_l = RF_{l-1} + (kernel_size_l - 1) * jump_{l-1}`` computed for the
   same layer stack with stride forced to 1 everywhere (variant A) versus the
   actual strided/pooled stack (variant B).
3. Does the network really "look at" its whole theoretical receptive field?
   -> a manual backward pass from a single deepest-layer unit, propagated
   through each layer's own ``backward()``, shows the *effective* receptive
   field is a soft, centrally-weighted blob -- much smaller than the
   theoretical box.

No training happens anywhere in this script: Conv2D keeps its He-initialized
random weights as-is, since the goal is showing network *structure*, not
accuracy.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from basicml.nn.module     import Module
from basicml.nn.conv       import Conv2D
from basicml.nn.pool       import MaxPool2D
from basicml.nn.activation import ReLU

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ---------------------------------------------------------------
SEED       = 0
IMAGE_SIZE = 32

# One layer stack, shared by the feature-map demo and both receptive-field
# variants below -- only the per-layer stride differs between variant A/B.
LAYER_SPECS = [
    {"kind": "conv", "kernel_size": 3, "padding": 1, "out_channels": 4},
    {"kind": "pool", "kernel_size": 2, "padding": 0},
    {"kind": "conv", "kernel_size": 3, "padding": 1, "out_channels": 8},
    {"kind": "conv", "kernel_size": 3, "padding": 1, "out_channels": 8},
    {"kind": "pool", "kernel_size": 2, "padding": 0},
]
STRIDES_NO_DOWNSAMPLE = [1, 1, 1, 1, 1]   # variant A: conv-only, stride 1
STRIDES_ACTUAL        = [1, 2, 1, 2, 2]   # variant B: stride/pooling included

MAX_CHANNELS_SHOWN = 8
FIG_SIZE           = (12, 3)
# ---------------------------------------------------------------------------


def make_sample_image(size: int = IMAGE_SIZE) -> np.ndarray:
    """Build one synthetic ``size x size`` grayscale image with clear edges.

    Combines a filled square, a diagonal stripe, and a hollow ring so later
    conv layers have unambiguous edges/corners to respond to, without
    needing an external image dataset.

    Args:
        size: Height and width of the square image, in pixels.

    Returns:
        Float64 array of shape ``(size, size)``, values in ``[0, 1]``.
    """
    rows, cols = np.mgrid[0:size, 0:size]
    image      = np.zeros((size, size), dtype=np.float64)

    square = (rows > size * 0.1) & (rows < size * 0.4) & \
             (cols > size * 0.1) & (cols < size * 0.4)
    image[square] = 1.0

    stripe = np.abs((rows - cols) - size * 0.1) < size * 0.03
    image[stripe] = 1.0

    center = size * 0.7
    radius = size * 0.18
    dist   = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    ring   = np.abs(dist - radius) < size * 0.04
    image[ring] = 1.0

    return image


def build_stack(strides: list[int], in_channels: int = 1,
                seed: int = SEED) -> tuple[list[Module], list[int]]:
    """Instantiate a flat list of ``basicml.nn`` modules from ``LAYER_SPECS``.

    A "conv" spec expands to ``[Conv2D, ReLU]``; a "pool" spec expands to a
    single ``MaxPool2D``. ``stage_ends`` marks, for each entry of
    ``LAYER_SPECS``, the index into the returned module list where that
    stage's output is available -- the granularity used for feature-map and
    receptive-field bookkeeping elsewhere in this script.

    Args:
        strides: Per-``LAYER_SPECS``-entry stride (same length as
            ``LAYER_SPECS``).
        in_channels: Number of channels of the input the stack will see.
        seed: Seed for ``Conv2D``'s He-initialized weights, for a
            reproducible run.

    Returns:
        Tuple ``(modules, stage_ends)``.
    """
    np.random.seed(seed)
    modules   : list[Module] = []
    stage_ends: list[int]    = []
    channels                 = in_channels

    for spec, stride in zip(LAYER_SPECS, strides):
        if spec["kind"] == "conv":
            modules.append(Conv2D(channels, int(spec["out_channels"]), int(spec["kernel_size"]),
                                  stride=stride, padding=int(spec["padding"]), init_type="he"))
            modules.append(ReLU())
            channels = int(spec["out_channels"])
        else:
            modules.append(MaxPool2D(int(spec["kernel_size"]), stride=stride, padding=int(spec["padding"])))
        stage_ends.append(len(modules) - 1)

    return modules, stage_ends


def forward_with_stage_outputs(modules: list[Module], x: np.ndarray,
                               stage_ends: list[int]) -> list[np.ndarray]:
    """Run ``x`` through ``modules`` in order, keeping each stage's output.

    Running the full module list (rather than stopping at each stage) is
    what leaves every layer's cache populated for the manual backward pass
    in Step 4.

    Args:
        modules: Flat module list, as returned by :func:`build_stack`.
        x: Input batch, shape ``(N, C, H, W)``.
        stage_ends: Index into ``modules`` marking each stage's last module.

    Returns:
        One output array per stage, in ``LAYER_SPECS`` order.
    """
    out          = x
    all_outputs  = []
    for module in modules:
        out = module(out)
        all_outputs.append(out)
    return [all_outputs[i] for i in stage_ends]


def plot_feature_maps(image: np.ndarray, stage_outputs: list[np.ndarray]) -> None:
    """Plot a grid of feature maps, one row per layer, input image included.

    Args:
        image: The original ``(H, W)`` input image.
        stage_outputs: One ``(1, C, H, W)`` array per layer, as returned by
            :func:`forward_with_stage_outputs`.
    """
    n_cols = min(MAX_CHANNELS_SHOWN, max(out.shape[1] for out in stage_outputs))
    n_rows = len(stage_outputs) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - CNN Feature Maps")

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_ylabel("input\n" f"{image.shape[0]}x{image.shape[1]}\n1ch", fontsize=8)
    for ax in axes[0, 1:]:
        ax.axis("off")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    for row, out in enumerate(stage_outputs, start=1):
        _, channels, height, width = out.shape
        shown = min(channels, n_cols)
        for col in range(n_cols):
            ax = axes[row, col]
            if col < shown:
                ax.imshow(out[0, col], cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
        axes[row, 0].set_ylabel(f"L{row}\n{height}x{width}\n{channels}ch", fontsize=8)

    fig.suptitle("Feature maps shrink and multiply in depth as the stack goes deeper")
    fig.tight_layout()


def compute_receptive_field(kernel_sizes: list[int], paddings: list[int],
                            strides: list[int], input_size: int) -> list[dict]:
    """Compute per-layer spatial size, jump, receptive field, and RF center.

    Implements the standard receptive-field recurrence used throughout this
    script: ``RF_l = RF_{l-1} + (kernel_size_l - 1) * jump_{l-1}``, where
    ``jump_{l-1}`` is the cumulative stride of every layer *before* layer
    ``l``. ``start`` tracks, in input-pixel units, where output unit 0's
    receptive field is centered -- output unit ``i``'s center is then
    ``start + i * jump``.

    Args:
        kernel_sizes: Kernel size of each layer, in order.
        paddings: Padding of each layer, in order (same length).
        strides: Stride of each layer, in order (same length).
        input_size: Height/width of the square input.

    Returns:
        One dict per layer with keys ``n`` (output spatial size), ``jump``,
        ``receptive_field``, ``start`` -- all describing state *after* that
        layer.
    """
    n, jump, rf, start = input_size, 1, 1, 0.5
    stats = []
    for k, p, s in zip(kernel_sizes, paddings, strides):
        n     = (n + 2 * p - k) // s + 1
        rf    = rf + (k - 1) * jump
        start = start + ((k - 1) / 2 - p) * jump
        jump  = jump * s
        stats.append({"n": n, "jump": jump, "receptive_field": rf, "start": start})
    return stats


def print_receptive_field_table(stats_a: list[dict], stats_b: list[dict]) -> None:
    """Print a layer-by-layer receptive field comparison for variants A/B.

    Args:
        stats_a: Per-layer stats for the stride-1 (no downsampling) variant.
        stats_b: Per-layer stats for the actual strided/pooled variant.
    """
    print("\nReceptive field growth: (A) stride-1, no downsampling  vs.  "
         "(B) actual stride/pooling")
    print(f"{'layer':>5} {'kernel':>6} {'RF (A)':>8} {'RF (B)':>8} "
         f"{'jump (A)':>9} {'jump (B)':>9}")
    for idx, (spec, a, b) in enumerate(zip(LAYER_SPECS, stats_a, stats_b), start=1):
        print(f"{idx:>5} {spec['kernel_size']:>6} {a['receptive_field']:>8} "
             f"{b['receptive_field']:>8} {a['jump']:>9} {b['jump']:>9}")
    print(f"-> variant A grows by a fixed amount each layer (linear); "
         f"variant B's jump compounds every stride, so its RF accelerates "
         f"(multiplicative) -- final RF: A={stats_a[-1]['receptive_field']}, "
         f"B={stats_b[-1]['receptive_field']}.\n")


def plot_receptive_field_growth(stats_a: list[dict], stats_b: list[dict]) -> None:
    """Plot theoretical receptive field size against layer depth for A/B.

    Args:
        stats_a: Per-layer stats for the stride-1 (no downsampling) variant.
        stats_b: Per-layer stats for the actual strided/pooled variant.
    """
    layers = list(range(len(LAYER_SPECS) + 1))
    rf_a   = [1] + [s["receptive_field"] for s in stats_a]
    rf_b   = [1] + [s["receptive_field"] for s in stats_b]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Receptive Field Growth")
    ax.plot(layers, rf_a, marker="o", label="A: stride 1, no downsampling (linear)")
    ax.plot(layers, rf_b, marker="o", label="B: actual stride/pooling (multiplicative)")
    ax.set_xlabel("layer depth")
    ax.set_ylabel("theoretical receptive field (px)")
    ax.set_title("Depth alone grows RF linearly -- stride/pooling grow it fast")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def effective_receptive_field(modules: list[Module], stage_outputs: list[np.ndarray],
                              image_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute the effective receptive field via a manual backward pass.

    Sets the gradient of a single unit -- center spatial position, the
    channel with the largest activation there (an untrained, randomly
    dead ReLU channel would otherwise zero the whole backward pass) -- in
    the deepest feature map to 1 and every other entry to 0, then
    propagates it back to the input by calling each already-cached
    module's own ``backward()`` in reverse order -- the same
    manual-gradient style used throughout ``basicml``, no autograd
    involved.

    Args:
        modules: Flat module list, already run forward (so every layer's
            cache is populated) by :func:`forward_with_stage_outputs`.
        stage_outputs: This run's per-stage outputs, used to locate the
            deepest feature map's spatial center.
        image_size: Height/width of the original input, for the returned
            heatmap's shape.

    Returns:
        Tuple ``(heatmap, center)`` -- the input-space gradient magnitude,
        shape ``(image_size, image_size)``, and the ``(row, col)`` output
        position whose gradient was seeded.
    """
    deepest              = stage_outputs[-1]
    _, _, out_h, out_w   = deepest.shape
    center               = (out_h // 2, out_w // 2)
    channel              = int(np.argmax(deepest[0, :, center[0], center[1]]))

    grad = np.zeros_like(deepest)
    grad[0, channel, center[0], center[1]] = 1.0

    for module in reversed(modules):
        grad = module.backward(grad)

    heatmap = np.abs(grad[0, 0])
    return heatmap, center


def plot_effective_vs_theoretical_rf(image: np.ndarray, heatmap: np.ndarray,
                                     stats_b: list[dict], center: tuple[int, int]) -> None:
    """Plot the raw effective-RF heatmap next to a theoretical-vs-effective overlay.

    The overlay panel draws the fixed-size theoretical receptive-field box
    from variant B (Step 3) on top of the effective-RF heatmap (Step 4), so
    it is visible at a glance that the effective RF is smaller and
    concentrated near the center rather than uniform across the box.

    Args:
        image: The original ``(H, W)`` input image.
        heatmap: Gradient-magnitude heatmap from
            :func:`effective_receptive_field`, shape matching ``image``.
        stats_b: Per-layer stats for the actual strided/pooled variant, used
            to place and size the theoretical RF box.
        center: The deepest-layer output position whose gradient was seeded.
    """
    final       = stats_b[-1]
    rf_size     = final["receptive_field"]
    center_row  = final["start"] + center[0] * final["jump"]
    center_col  = final["start"] + center[1] * final["jump"]

    fig, (ax_heat, ax_overlay) = plt.subplots(1, 2, figsize=(11, 5.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Effective vs Theoretical Receptive Field")

    ax_heat.imshow(heatmap, cmap="hot")
    ax_heat.set_title("Effective RF\n(|gradient| of one deep unit w.r.t. the input)")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])

    ax_overlay.imshow(image, cmap="gray")
    ax_overlay.imshow(heatmap, cmap="hot", alpha=0.6)
    box = Rectangle((center_col - rf_size / 2, center_row - rf_size / 2), rf_size, rf_size,
                    edgecolor="cyan", facecolor="none", linewidth=2,
                    label=f"theoretical RF ({rf_size}x{rf_size}px)")
    ax_overlay.add_patch(box)
    ax_overlay.set_title("Theoretical box vs. effective RF\n(box is uniform -- gradient is not)")
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    ax_overlay.legend(loc="upper right", fontsize=8)

    fig.tight_layout()


def main() -> None:
    """Run the full feature-map / receptive-field demo end to end."""
    image = make_sample_image()
    x     = image[None, None, :, :]

    modules, stage_ends = build_stack(STRIDES_ACTUAL)
    stage_outputs        = forward_with_stage_outputs(modules, x, stage_ends)
    plot_feature_maps(image, stage_outputs)

    kernel_sizes = [int(spec["kernel_size"]) for spec in LAYER_SPECS]
    paddings     = [int(spec["padding"]) for spec in LAYER_SPECS]
    stats_a      = compute_receptive_field(kernel_sizes, paddings, STRIDES_NO_DOWNSAMPLE, IMAGE_SIZE)
    stats_b      = compute_receptive_field(kernel_sizes, paddings, STRIDES_ACTUAL, IMAGE_SIZE)
    print_receptive_field_table(stats_a, stats_b)
    plot_receptive_field_growth(stats_a, stats_b)

    heatmap, center = effective_receptive_field(modules, stage_outputs, IMAGE_SIZE)
    plot_effective_vs_theoretical_rf(image, heatmap, stats_b, center)

    plt.show()


if __name__ == "__main__":
    main()
