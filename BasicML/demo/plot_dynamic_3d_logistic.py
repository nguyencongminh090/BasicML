# AI generated (refactored/authored with Claude Code)
"""Animated 3D cost surface for logistic regression on 1D data.

Run directly: python BasicML/demo/plot_dynamic_3d_logistic.py

A single panel: the BCE cost surface over ``(w, b)`` space with the Adam
optimizer's path traced across it as training progresses.
"""
import os
import sys
import random
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.nn.sequential  import Sequential
from basicml.nn.linear      import Linear
from basicml.nn.activation  import Sigmoid
from basicml.nn.loss        import BinaryCrossEntropy
from basicml.optim.adam     import Adam

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED           = None
N_SAMPLES      = 100
X_RANGE        = (0.0, 10.0)
LABEL_FRACTION = 0.25

INIT_W         = -1.0
INIT_B         = -2.0

EPOCHS         = 1500
LR_CYCLE       = (0.05, 0.4, 0.01)    # (base, peak, final)
WARMUP_FRAC    = 0.3

GRID_RESOLUTION = 50
GRID_MARGIN     = 7.5                  # (w, b) padding around the trajectory
Z_HEADROOM      = 15.0                 # upper clip of the cost axis
FRAME_INTERVAL  = 10
FIG_SIZE        = (10, 8)
# ----------------------------------------------------------------------


@dataclass
class OneCycle:
    """One-cycle learning-rate schedule: a cosine warmup then a cosine anneal.

    Attributes:
        lr: ``(base, peak, final)`` learning rates.
        warmup_frac: Fraction of training spent in the warmup phase.
    """
    lr:          tuple[float, float, float]
    warmup_frac: float

    def at(self, progress: float) -> float:
        """Return the learning rate at a given point in training.

        Args:
            progress: Training progress in ``[0, 1]``.

        Returns:
            The learning rate for this step.
        """
        lr_base, lr_peak, lr_final = self.lr

        if progress < self.warmup_frac:
            phase  = progress / self.warmup_frac
            factor = 0.5 * (1 - np.cos(np.pi * phase))
            return lr_base + (lr_peak - lr_base) * factor

        phase  = (progress - self.warmup_frac) / (1.0 - self.warmup_frac)
        factor = 0.5 * (1 + np.cos(np.pi * phase))
        return lr_final + (lr_peak - lr_final) * factor


@dataclass
class TrainingHistory:
    """Per-epoch trajectory recorded during training.

    Attributes:
        weight: Weight value at each epoch, shape ``(EPOCHS,)``.
        bias: Bias value at each epoch, shape ``(EPOCHS,)``.
        cost: BCE cost at each epoch, shape ``(EPOCHS,)``.
    """
    weight: np.ndarray
    bias:   np.ndarray
    cost:   np.ndarray


def make_threshold_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate a standardized 1D threshold classification dataset.

    Returns:
        Tuple ``(x, y)`` of float64 arrays shaped ``(N_SAMPLES, 1)``; ``x`` is
        standardized to zero mean and unit variance.
    """
    rng       = random.Random(SEED)
    lo, hi    = X_RANGE
    threshold = lo + LABEL_FRACTION * (hi - lo)

    x = np.array([rng.uniform(lo, hi) for _ in range(N_SAMPLES)]).reshape(-1, 1)
    y = (x > threshold).astype(np.float64)
    x = (x - x.mean()) / x.std()
    return x, y


def train_and_record(x: np.ndarray, y: np.ndarray) -> TrainingHistory:
    """Train a ``Linear + Sigmoid`` model and record its ``(w, b, cost)`` path.

    The linear layer is forced to start at ``(INIT_W, INIT_B)`` so the animated
    path always begins from the same corner of the surface.

    Args:
        x: Input features, shape ``(N_SAMPLES, 1)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.

    Returns:
        A :class:`TrainingHistory` with one entry per epoch.
    """
    linear = Linear(in_features=1, out_features=1)
    assert linear.b is not None
    linear.w.data = np.array([[INIT_W]])
    linear.b.data = np.array([[INIT_B]])

    model     = Sequential(linear, Sigmoid())
    criterion = BinaryCrossEntropy()
    optimizer = Adam(model.parameters(), lr=LR_CYCLE[0])
    schedule  = OneCycle(LR_CYCLE, WARMUP_FRAC)

    weight_hist: list[float] = []
    bias_hist:   list[float] = []
    cost_hist:   list[float] = []

    print("Training model to gather history...")
    for epoch in range(EPOCHS):
        optimizer.lr = schedule.at(epoch / EPOCHS)

        weight_hist.append(linear.w.data[0, 0])
        bias_hist.append(linear.b.data[0, 0])

        cost = criterion(model(x), y)
        cost_hist.append(cost)

        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

    print(f"Training complete. Final cost: {cost_hist[-1]:.4f}")
    return TrainingHistory(
        weight=np.array(weight_hist),
        bias=np.array(bias_hist),
        cost=np.array(cost_hist),
    )


def bce_cost_surface(x: np.ndarray, y: np.ndarray,
                     w_range: tuple[float, float],
                     b_range: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Evaluate the BCE cost on a ``(w, b)`` grid.

    For each grid point the sigmoid predictions over the whole dataset are
    formed and the mean binary cross-entropy is computed (probabilities are
    clipped to avoid ``log(0)``).

    Args:
        x: Input features, shape ``(N_SAMPLES, 1)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.
        w_range: ``(min, max)`` weight range for the grid.
        b_range: ``(min, max)`` bias range for the grid.

    Returns:
        Tuple ``(w_grid, b_grid, z_grid)``, each shaped
        ``(GRID_RESOLUTION, GRID_RESOLUTION)``.
    """
    w_vals = np.linspace(*w_range, GRID_RESOLUTION)
    b_vals = np.linspace(*b_range, GRID_RESOLUTION)
    w_grid, b_grid = np.meshgrid(w_vals, b_vals)

    logits = w_grid[..., None, None] * x + b_grid[..., None, None]
    probs  = np.clip(1 / (1 + np.exp(-logits)), 1e-15, 1 - 1e-15)
    z_grid = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs), axis=(-2, -1))
    return w_grid, b_grid, z_grid


def animate(x: np.ndarray, y: np.ndarray, history: TrainingHistory) -> FuncAnimation:
    """Draw the 3D cost surface and animate the optimizer path over it.

    Args:
        x: Input features, shape ``(N_SAMPLES, 1)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.
        history: Recorded ``(w, b, cost)`` trajectory to animate.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation` handle (kept alive so
        the animation is not garbage-collected).
    """
    w_opt, b_opt = history.weight[-1], history.bias[-1]
    min_cost     = history.cost[-1]

    w_margin = max(float(np.ptp(history.weight)), GRID_MARGIN) * 1.5
    b_margin = max(float(np.ptp(history.bias)), GRID_MARGIN) * 1.5
    w_range  = (history.weight.min() - w_margin, history.weight.max() + w_margin)
    b_range  = (history.bias.min() - b_margin, history.bias.max() + b_margin)

    w_grid, b_grid, z_grid = bce_cost_surface(x, y, w_range, b_range)
    z_grid = np.clip(z_grid, 0, min_cost + Z_HEADROOM)

    fig = plt.figure(figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Logistic Regression 3D Cost Surface")

    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(w_grid, b_grid, z_grid, cmap="viridis", alpha=0.6, edgecolor="none")
    path_line_3d, = ax.plot([], [], [], color="black", marker="o", markersize=3,
                            linewidth=2, label="Optimizer Path")
    ax.plot([w_opt], [b_opt], [min_cost], marker="*", color="red", markersize=12, label="End")
    ax.set_box_aspect((2, 2, 1))
    ax.set_title("3D Gradient Path on Cost Surface (Logistic Regression)")
    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Bias (b)")
    ax.set_zlabel("Cost (BCE)")
    ax.view_init(elev=30, azim=-60)
    ax.legend()

    def update(frame: int):
        """Extend the traced path to include epoch ``frame``."""
        path_line_3d.set_data(history.weight[:frame + 1], history.bias[:frame + 1])
        path_line_3d.set_3d_properties(history.cost[:frame + 1])
        return path_line_3d,

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=len(history.cost),
                         interval=FRAME_INTERVAL, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    """Generate data, train while recording history, then show the animation."""
    x, y    = make_threshold_dataset()
    history = train_and_record(x, y)
    animate(x, y, history)


if __name__ == "__main__":
    main()
