# AI generated (refactored/authored with Claude Code)
"""Static figure: how one hidden layer transforms the feature space.

Run directly: python BasicML/demo/plot_layer_transformations.py

Eight panels: the data points after each layer (row 1) and the coordinate grid
deformed by each layer (row 2, Christopher Olah style). Hidden spaces with more
than 2 dimensions are projected down to 2D with PCA.
"""
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from basicml.datasets       import make_moons
from basicml.nn.sequential  import Sequential
from basicml.nn.linear      import Linear
from basicml.nn.activation  import Tanh, Sigmoid
from basicml.nn.loss        import BinaryCrossEntropy
from basicml.optim.momentum import Momentum
from basicml.visualize      import plot_decision_boundary

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED        = 42
N_SAMPLES   = 200
NOISE       = 0.15

HIDDEN_DIM  = 5
LEARN_RATE  = 0.08
MOMENTUM    = 0.92
EPOCHS      = 5000

GRID_LINES  = 25                       # grid lines per direction
GRID_POINTS = 200                      # points sampled along each grid line
X_LINE_SPAN = (-1.5, 2.5)
Y_LINE_SPAN = (-1.0, 1.5)

FIG_SIZE    = (18, 10)
# ----------------------------------------------------------------------


def project_to_2d(data: np.ndarray,
                  basis: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Project N-D data to 2D via PCA (SVD of the centered data).

    Args:
        data: Array of shape ``(N, D)``.
        basis: Optional pre-computed ``(D, 2)`` projection basis; when omitted
            and ``D > 2``, the top-2 principal components are used.

    Returns:
        Tuple ``(data_2d, basis)`` where ``data_2d`` is ``(N, 2)``.
    """
    if data.shape[1] == 2:
        return data, np.eye(2)

    if basis is None:
        centered   = data - data.mean(axis=0, keepdims=True)
        _, _, vt   = np.linalg.svd(centered, full_matrices=False)
        basis      = vt[:2].T          # shape (D, 2)

    return data @ basis, basis


def build_and_train(x: np.ndarray, y: np.ndarray):
    """Build and train the ``Linear -> Tanh -> Linear -> Sigmoid`` model.

    Args:
        x: Input features, shape ``(N_SAMPLES, 2)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.

    Returns:
        Tuple ``(model, linear_1, activation_1, linear_2, activation_2)`` so the
        caller can probe individual layers.
    """
    linear_1     = Linear(in_features=2,          out_features=HIDDEN_DIM, init_type="he")
    activation_1 = Tanh()
    linear_2     = Linear(in_features=HIDDEN_DIM, out_features=1,          init_type="xavier")
    activation_2 = Sigmoid()

    model     = Sequential(linear_1, activation_1, linear_2, activation_2)
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    cost = float("inf")
    for _ in range(EPOCHS):
        cost = criterion(model(x), y)
        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

    print(f"Model (hidden dim {HIDDEN_DIM}) trained. Final loss: {cost:.4f}")
    return model, linear_1, activation_1, linear_2, activation_2


def make_grid_lines() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build the flat input-space coordinate grid.

    Returns:
        Tuple ``(horizontal, vertical)`` of polyline lists spanning
        ``X_LINE_SPAN`` x ``Y_LINE_SPAN``.
    """
    x_lo, x_hi = X_LINE_SPAN
    y_lo, y_hi = Y_LINE_SPAN

    horizontal = [np.column_stack([np.linspace(x_lo, x_hi, GRID_POINTS),
                                   np.full(GRID_POINTS, y_val)])
                  for y_val in np.linspace(y_lo, y_hi, GRID_LINES)]
    vertical   = [np.column_stack([np.full(GRID_POINTS, x_val),
                                   np.linspace(y_lo, y_hi, GRID_POINTS)])
                  for x_val in np.linspace(x_lo, x_hi, GRID_LINES)]
    return horizontal, vertical


def plot_layer_transformations() -> None:
    """Train the model and render the full 8-panel transformation figure."""
    np.random.seed(SEED)
    x, y   = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    labels = y.ravel()

    model, linear_1, activation_1, linear_2, _ = build_and_train(x, y)

    # Per-layer representation of the data ------------------------------
    z1_raw = linear_1.forward(x)             # (N, HIDDEN_DIM)
    h1_raw = activation_1.forward(z1_raw)    # (N, HIDDEN_DIM)
    z2     = linear_2.forward(h1_raw)        # (N, 1)
    y_hat  = 1 / (1 + np.exp(-z2))           # (N, 1)

    z1, basis_z1 = project_to_2d(z1_raw)
    h1, basis_h1 = project_to_2d(h1_raw)

    assert linear_2.b is not None
    w2_projected = linear_2.w.data.ravel() @ basis_h1   # (2,)
    b2           = float(linear_2.b.data.ravel()[0])

    horizontal_grid, vertical_grid = make_grid_lines()

    fig = plt.figure(figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"BasicML - Hidden Space Transformation (Hidden Dim: {HIDDEN_DIM})")

    projected = HIDDEN_DIM > 2
    dim_label = f"({HIDDEN_DIM}D $\\to$ 2D PCA)" if projected else "(2D Space)"

    # ROW 1: data points after each layer ------------------------------
    ax = fig.add_subplot(2, 4, 1)
    plot_decision_boundary(model, x, y, ax=ax,
                           title="1. Input Space (X)\nNon-linear Problem", show=False)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax = fig.add_subplot(2, 4, 2)
    ax.scatter(z1[:, 0], z1[:, 1], c=labels, cmap="Spectral", edgecolors="k", s=30, alpha=0.9)
    ax.set_title(f"2. After Linear 1 ($Z_1$) {dim_label}\nRotation, Shearing & Translation")
    ax.set_xlabel("Dim 1 (PCA Proj)" if projected else "$z_{1,1}$")
    ax.set_ylabel("Dim 2 (PCA Proj)" if projected else "$z_{1,2}$")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = fig.add_subplot(2, 4, 3)
    ax.scatter(h1[:, 0], h1[:, 1], c=labels, cmap="Spectral", edgecolors="k", s=30, alpha=0.9)
    if abs(w2_projected[1]) > 1e-6:
        h_axis   = np.linspace(-1.5, 1.5, 100)
        boundary = -(w2_projected[0] * h_axis + b2) / w2_projected[1]
        visible  = (boundary >= -2.0) & (boundary <= 2.0)
        ax.plot(h_axis[visible], boundary[visible], color="black", linewidth=2.5,
                linestyle="--", label="Separating Boundary")
        ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"3. Hidden Space ($H_1$) {dim_label}\nDisentangled -> Linearly Separable!")
    ax.set_xlabel("Dim 1 (PCA Proj)" if projected else "$h_{1,1}$")
    ax.set_ylabel("Dim 2 (PCA Proj)" if projected else "$h_{1,2}$")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = fig.add_subplot(2, 4, 4)
    ax.scatter(range(len(y_hat)), y_hat, c=labels, cmap="Spectral", edgecolors="k", s=25, alpha=0.8)
    ax.axhline(0.5, color="black", linestyle="--", label="Threshold 0.5")
    ax.set_title(r"4. Output Space ($\hat{y} = \sigma(Z_2)$)" + "\nFinal Prediction Probabilities")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel(r"Probability $\hat{y}$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    # ROW 2: coordinate grid deformation ------------------------------
    ax = fig.add_subplot(2, 4, 5)
    for pts in horizontal_grid:
        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", alpha=0.4, linewidth=0.8)
    for pts in vertical_grid:
        ax.plot(pts[:, 0], pts[:, 1], color="coral", alpha=0.4, linewidth=0.8)
    ax.set_title("5. Input Grid (Flat Euclidean)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(*X_LINE_SPAN)
    ax.set_ylim(*Y_LINE_SPAN)

    ax = fig.add_subplot(2, 4, 6)
    for pts in horizontal_grid:
        warped = linear_1.forward(pts) @ basis_z1
        ax.plot(warped[:, 0], warped[:, 1], color="steelblue", alpha=0.4, linewidth=0.8)
    for pts in vertical_grid:
        warped = linear_1.forward(pts) @ basis_z1
        ax.plot(warped[:, 0], warped[:, 1], color="coral", alpha=0.4, linewidth=0.8)
    ax.set_title(f"6. Grid after Linear 1 {dim_label}\nAffine Warped")
    ax.set_xlabel("Dim 1 (PCA)" if projected else "$z_{1,1}$")
    ax.set_ylabel("Dim 2 (PCA)" if projected else "$z_{1,2}$")

    ax = fig.add_subplot(2, 4, 7)
    for pts in horizontal_grid:
        warped = activation_1.forward(linear_1.forward(pts)) @ basis_h1
        ax.plot(warped[:, 0], warped[:, 1], color="steelblue", alpha=0.4, linewidth=0.8)
    for pts in vertical_grid:
        warped = activation_1.forward(linear_1.forward(pts)) @ basis_h1
        ax.plot(warped[:, 0], warped[:, 1], color="coral", alpha=0.4, linewidth=0.8)
    ax.set_title(f"7. Grid after Tanh {dim_label}\nManifold Folded")
    ax.set_xlabel("Dim 1 (PCA)" if projected else "$h_{1,1}$")
    ax.set_ylabel("Dim 2 (PCA)" if projected else "$h_{1,2}$")

    ax = fig.add_subplot(2, 4, 8)
    ax.axis("off")
    ax.text(0.05, 0.5,
            f"Deep Network ({HIDDEN_DIM} Hidden Neurons):\n\n"
            "1. Input Space (2D):\n"
            "   Non-linearly entangled data.\n\n"
            f"2. Hidden Layer ({HIDDEN_DIM}D):\n"
            f"   Projected from 2D -> {HIDDEN_DIM}D.\n"
            "   Extra dimensions give the network\n"
            "   room to unfold the data manifold.\n\n"
            "3. Visualization via PCA:\n"
            f"   {HIDDEN_DIM}D space projected to 2D\n"
            "   along top 2 principal components,\n"
            "   showing clear linear separability!",
            fontsize=10, verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="linen", edgecolor="gray", alpha=0.8))

    plt.suptitle(f"BasicML - Layer Space Transformation (Hidden Neurons: {HIDDEN_DIM})",
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_layer_transformations()
