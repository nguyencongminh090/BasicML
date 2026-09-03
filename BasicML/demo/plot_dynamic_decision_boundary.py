# AI generated (refactored/authored with Claude Code)
"""Animated decision boundary of a small MLP on the two-moons dataset.

Run directly: python BasicML/demo/plot_dynamic_decision_boundary.py

Left panel: the decision region as it evolves per epoch. Right panel: the
learning curve.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.datasets       import make_moons
from basicml.nn.sequential  import Sequential
from basicml.nn.linear      import Linear
from basicml.nn.activation  import Tanh, Sigmoid, ReLU
from basicml.nn.loss        import BinaryCrossEntropy
from basicml.optim.adam     import Adam

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED           = 45
N_SAMPLES      = 200
NOISE          = 0.20

HIDDEN_DIM     = 4
LEARN_RATE     = 0.01
EPOCHS         = 2000
RECORD_EVERY   = 6                      # record a history frame every N epochs

GRID_STEP      = 0.03                   # decision-boundary grid resolution
GRID_PADDING   = 0.5
FRAME_INTERVAL = 5
FIG_SIZE       = (14, 6)
# ----------------------------------------------------------------------


def build_model() -> Sequential:
    """Build the 4-layer MLP (ReLU / Sigmoid / ReLU / Sigmoid) used in the demo.

    Returns:
        A :class:`Sequential` mapping 2 input features to a single probability.
    """
    return Sequential(
        Linear(in_features=2,          out_features=HIDDEN_DIM, init_type="he"),
        ReLU(),
        Linear(in_features=HIDDEN_DIM, out_features=HIDDEN_DIM, init_type="xavier"),
        Sigmoid(),
        Linear(in_features=HIDDEN_DIM, out_features=HIDDEN_DIM, init_type="he"),
        ReLU(),
        Linear(in_features=HIDDEN_DIM, out_features=1,          init_type="xavier"),
        Sigmoid(),
    )


def make_grid(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an evaluation mesh covering the data with ``GRID_PADDING`` margin.

    Args:
        x: Input features, shape ``(n_samples, 2)``.

    Returns:
        Tuple ``(xx, yy, grid_points)`` where ``xx``/``yy`` are the meshgrid
        arrays and ``grid_points`` is their flattened ``(n_grid, 2)`` form.
    """
    x_min, x_max = x[:, 0].min() - GRID_PADDING, x[:, 0].max() + GRID_PADDING
    y_min, y_max = x[:, 1].min() - GRID_PADDING, x[:, 1].max() + GRID_PADDING
    xx, yy = np.meshgrid(np.arange(x_min, x_max, GRID_STEP),
                         np.arange(y_min, y_max, GRID_STEP))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid_points


def train_and_record(model: Sequential, x: np.ndarray, y: np.ndarray):
    """Train ``model`` and snapshot its decision region every ``RECORD_EVERY`` epochs.

    Args:
        model: The MLP to train (updated in place).
        x: Input features, shape ``(N_SAMPLES, 2)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.

    Returns:
        Tuple ``(xx, yy, epochs_seen, cost_hist, accuracy_hist, region_hist)``
        holding the mesh plus per-frame epoch index, loss, accuracy (%), and
        the ``0/1`` region map shaped like ``xx``.
    """
    criterion = BinaryCrossEntropy()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)
    xx, yy, grid_points = make_grid(x)

    epochs_seen  : list[int]        = []
    cost_hist    : list[float]      = []
    accuracy_hist: list[float]      = []
    region_hist  : list[np.ndarray] = []

    print("Training model and collecting history for animation...")
    for epoch in range(EPOCHS):
        y_pred = model(x)
        cost   = criterion(y_pred, y)

        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

        if epoch % RECORD_EVERY == 0 or epoch == EPOCHS - 1:
            epochs_seen.append(epoch)
            cost_hist.append(cost)

            regions = (model(grid_points) >= 0.5).astype(int).reshape(xx.shape)
            region_hist.append(regions)

            accuracy = float(np.mean((y_pred >= 0.5).astype(int) == y)) * 100.0
            accuracy_hist.append(accuracy)

    print(f"Training finished. Final loss: {cost_hist[-1]:.4f}, "
          f"final acc: {accuracy_hist[-1]:.1f}%")
    return xx, yy, epochs_seen, cost_hist, accuracy_hist, region_hist


def animate(x, y, xx, yy, epochs_seen, cost_hist, accuracy_hist, region_hist) -> FuncAnimation:
    """Animate the decision region (left) and the learning curve (right).

    Args:
        x: Input features, shape ``(N_SAMPLES, 2)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.
        xx: Meshgrid x-coordinates.
        yy: Meshgrid y-coordinates.
        epochs_seen: Epoch index for each recorded frame.
        cost_hist: BCE loss for each frame.
        accuracy_hist: Training accuracy (%) for each frame.
        region_hist: ``0/1`` region map for each frame, shaped like ``xx``.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation` handle.
    """
    fig, (ax_boundary, ax_curve) = plt.subplots(1, 2, figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Dynamic Decision Boundary Animation")

    labels = y.ravel()

    cost_line, = ax_curve.plot([], [], color="seagreen", linewidth=2, label="BCE Loss")
    ax_curve.set_xlim(0, EPOCHS)
    ax_curve.set_ylim(0, max(cost_hist) * 1.1)
    ax_curve.set_title("Learning Curve (Loss vs Epochs)")
    ax_curve.set_xlabel("Epochs")
    ax_curve.set_ylabel("Binary Cross Entropy Loss")
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.legend(loc="upper right")

    def update(frame: int):
        """Redraw the decision region and extend the loss curve for ``frame``."""
        ax_boundary.clear()
        regions = region_hist[frame]
        ax_boundary.contourf(xx, yy, regions, cmap="Spectral", alpha=0.75)
        ax_boundary.contour(xx, yy, regions, levels=[0.5], colors="black",
                            linewidths=1.5, alpha=0.8)
        ax_boundary.scatter(x[:, 0], x[:, 1], c=labels, cmap="Spectral",
                            edgecolors="k", s=35, alpha=0.9)
        ax_boundary.set_xlim(xx.min(), xx.max())
        ax_boundary.set_ylim(yy.min(), yy.max())
        ax_boundary.set_title(f"Decision Boundary (Epoch {epochs_seen[frame]:4d}) | "
                              f"Acc: {accuracy_hist[frame]:.1f}%")
        ax_boundary.set_xlabel("Feature $x_1$")
        ax_boundary.set_ylabel("Feature $x_2$")

        cost_line.set_data(epochs_seen[:frame + 1], cost_hist[:frame + 1])
        ax_curve.set_title(f"Learning Curve: Loss = {cost_hist[frame]:.4f}")
        return cost_line,

    print("Rendering animation...")
    anim = FuncAnimation(fig, update, frames=len(epochs_seen),
                         interval=FRAME_INTERVAL, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    """Generate two-moons data, train while recording, then show the animation."""
    np.random.seed(SEED)
    x, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    model = build_model()
    animate(x, y, *train_and_record(model, x, y))


if __name__ == "__main__":
    main()
