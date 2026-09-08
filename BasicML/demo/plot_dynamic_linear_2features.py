# AI generated (refactored/authored with Claude Code)
"""Animated linear-regression training on a synthetic 2-feature dataset.

Run directly: python BasicML/demo/plot_dynamic_linear_2features.py

The dataset has two features (x1, x2) whose point cloud scatters around a
parabola in the (x1, x2) plane; the target y is a noisy linear combination of
both features. Five panels: the 3D fit, the learning curve, cost-vs-w1, and
the gradient descent path over (w1, w2) (2D contour + 3D surface). A plain
Momentum optimizer with a fixed learning rate drives it, matching
plot_dynamic_linear.py.
"""
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.datasets.synthetic import make_parabola_regression
from basicml.nn.linear      import Linear
from basicml.nn.loss        import MSELoss
from basicml.optim.momentum import Momentum
from basicml.optim.sgd      import SGD

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
N_SAMPLES     = 200
X1_RANGE      = (-3.0, 3.0)
CURVATURE     = 1.0
FEATURE_NOISE = 0.4
TRUE_WEIGHTS  = (2.0, -1.0)
TRUE_BIAS     = 1.0
TARGET_NOISE  = 0.6
RANDOM_STATE  = 0

INIT_W1     = -3.0
INIT_W2     = 2.0
INIT_B      = -2.0

EPOCHS      = 400
LEARN_RATE  = 0.01

GRID_RESOLUTION = 50
FRAME_INTERVAL  = 5                    # ms between frames
FIG_SIZE        = (18, 10)
# ----------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Per-epoch trajectory recorded during training.

    Attributes:
        w1: First weight value at each epoch.
        w2: Second weight value at each epoch.
        bias: Bias value at each epoch.
        cost: MSE cost at each epoch.
    """
    w1:   np.ndarray
    w2:   np.ndarray
    bias: np.ndarray
    cost: np.ndarray


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate the 2-feature parabola-scattered regression dataset.

    Returns:
        Tuple ``(x, y)``: ``x`` shaped ``(n_samples, 2)`` with columns
        ``(x1, x2)``, ``y`` shaped ``(n_samples, 1)``.
    """
    return make_parabola_regression(
        n_samples=N_SAMPLES,
        x1_range=X1_RANGE,
        curvature=CURVATURE,
        feature_noise=FEATURE_NOISE,
        weights=TRUE_WEIGHTS,
        bias=TRUE_BIAS,
        target_noise=TARGET_NOISE,
        random_state=RANDOM_STATE,
    )


def train_and_record(x: np.ndarray, y: np.ndarray) -> TrainingHistory:
    """Train a ``Linear`` model and record its ``(w1, w2, b, cost)`` path.

    Starts from ``(INIT_W1, INIT_W2, INIT_B)`` and runs ``EPOCHS`` full-batch
    Momentum steps at a fixed ``LEARN_RATE``.

    Args:
        x: Input features, shape ``(n_samples, 2)``.
        y: Targets, shape ``(n_samples, 1)``.

    Returns:
        A :class:`TrainingHistory` with one entry per epoch.
    """
    model = Linear(in_features=2, out_features=1)
    assert model.b is not None
    model.w.data = np.array([[INIT_W1], [INIT_W2]])
    model.b.data = np.array([[INIT_B]])

    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=LEARN_RATE)

    w1_hist:   list[float] = []
    w2_hist:   list[float] = []
    bias_hist: list[float] = []
    cost_hist: list[float] = []

    print("Training model to gather history...")
    for _ in range(EPOCHS):
        w1_hist.append(model.w.data[0, 0])
        w2_hist.append(model.w.data[1, 0])
        bias_hist.append(model.b.data[0, 0])

        cost = criterion(model(x), y)
        cost_hist.append(cost)

        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

    print(f"Training complete. Final cost: {cost_hist[-1]:.4f}")
    return TrainingHistory(
        w1=np.array(w1_hist),
        w2=np.array(w2_hist),
        bias=np.array(bias_hist),
        cost=np.array(cost_hist),
    )


def closed_form_optimum(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return the least-squares optimum ``(w1, w2, b, cost)`` via the normal equation.

    Args:
        x: Input features, shape ``(n_samples, 2)``.
        y: Targets, shape ``(n_samples, 1)``.

    Returns:
        Tuple ``(w1, w2, b, mse)`` at the global minimum.
    """
    design = np.hstack([x, np.ones((x.shape[0], 1))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    w1, w2, b = beta.ravel()
    cost = float(np.mean((design @ beta - y) ** 2))
    return float(w1), float(w2), float(b), cost


def cost_surface(x: np.ndarray, y: np.ndarray, b: float,
                 w1_range: tuple[float, float],
                 w2_range: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Evaluate the MSE cost on a ``(w1, w2)`` grid with the bias held fixed.

    Args:
        x: Input features, shape ``(n_samples, 2)``.
        y: Targets, shape ``(n_samples, 1)``.
        b: Bias value held constant while sweeping the weights.
        w1_range: ``(min, max)`` range for the first weight.
        w2_range: ``(min, max)`` range for the second weight.

    Returns:
        Tuple ``(w1_grid, w2_grid, z_grid)``, each shaped
        ``(GRID_RESOLUTION, GRID_RESOLUTION)``.
    """
    w1_vals = np.linspace(*w1_range, GRID_RESOLUTION)
    w2_vals = np.linspace(*w2_range, GRID_RESOLUTION)
    w1_grid, w2_grid = np.meshgrid(w1_vals, w2_vals)

    x1, x2 = x[:, 0], x[:, 1]
    preds  = (w1_grid[..., None] * x1 + w2_grid[..., None] * x2 + b)
    z_grid = np.mean((preds - y.ravel()) ** 2, axis=-1)
    return w1_grid, w2_grid, z_grid


def animate(x: np.ndarray, y: np.ndarray, history: TrainingHistory) -> FuncAnimation:
    """Build the 5-panel figure and animate it over the recorded history.

    Args:
        x: Input features, shape ``(n_samples, 2)``.
        y: Targets, shape ``(n_samples, 1)``.
        history: Recorded ``(w1, w2, b, cost)`` trajectory to animate.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation` handle.
    """
    w1_opt, w2_opt, b_opt, min_cost = closed_form_optimum(x, y)

    w1_margin = max(float(np.ptp(history.w1)), 2.0) * 0.4
    w2_margin = max(float(np.ptp(history.w2)), 2.0) * 0.4
    w1_range  = (min(history.w1.min(), w1_opt) - w1_margin,
                 max(history.w1.max(), w1_opt) + w1_margin)
    w2_range  = (min(history.w2.min(), w2_opt) - w2_margin,
                 max(history.w2.max(), w2_opt) + w2_margin)

    w1_grid, w2_grid, z_grid = cost_surface(x, y, b_opt, w1_range, w2_range)

    x1, x2 = x[:, 0], x[:, 1]

    fig = plt.figure(figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            "BasicML - Linear Regression (2 Features) Dynamic Training")

    ax_fit = fig.add_subplot(231, projection="3d")
    ax_fit.scatter(x1, x2, y.ravel(), color="blue", alpha=0.6, label="Training Data")
    fit_x1, fit_x2 = np.meshgrid(np.linspace(x1.min(), x1.max(), 10),
                                 np.linspace(x2.min(), x2.max(), 10))
    fit_plane = ax_fit.plot_surface(fit_x1, fit_x2,
                                    np.zeros_like(fit_x1),
                                    color="red", alpha=0.3)
    ax_fit.set_title("1. Linear Regression Fit")
    ax_fit.set_xlabel("x1")
    ax_fit.set_ylabel("x2")
    ax_fit.set_zlabel("y")
    ax_fit.legend()

    ax_curve = fig.add_subplot(232)
    cost_line, = ax_curve.plot([], [], color="green", linewidth=2, label="MSE Loss")
    ax_curve.set_xlim(0, len(history.cost))
    ax_curve.set_ylim(0, history.cost.max() * 1.1)
    ax_curve.set_title("2. Learning Curve")
    ax_curve.set_xlabel("Epochs")
    ax_curve.set_ylabel("Cost (MSE)")
    ax_curve.legend()
    ax_curve.grid(True, linestyle="--", alpha=0.6)

    ax_costw = fig.add_subplot(233)
    costw_line, = ax_costw.plot([], [], color="purple", linewidth=2, label="Cost vs w1")
    ax_costw.set_xlim(*w1_range)
    ax_costw.set_ylim(0, history.cost.max() * 1.1)
    ax_costw.set_title("5. Cost vs Weight (w1)")
    ax_costw.set_xlabel("Weight (w1)")
    ax_costw.set_ylabel("Cost (MSE)")
    ax_costw.legend()
    ax_costw.grid(True, linestyle="--", alpha=0.6)

    ax_path = fig.add_subplot(234)
    contour = ax_path.contour(w1_grid, w2_grid, z_grid,
                              levels=np.linspace(min_cost, z_grid.max(), 20),
                              cmap="viridis", alpha=0.8)
    ax_path.clabel(contour, inline=True, fontsize=8)
    ax_path.plot([w1_opt], [w2_opt], marker="*", color="red", markersize=12,
                 label=f"Global Min ({w1_opt:.2f}, {w2_opt:.2f})")
    path_line, = ax_path.plot([], [], color="black", marker="o", markersize=3,
                              linewidth=1, alpha=0.7, label="Optimizer Path")
    ax_path.set_xlim(*w1_range)
    ax_path.set_ylim(*w2_range)
    ax_path.set_title("3. 2D Gradient Path on Cost Surface")
    ax_path.set_xlabel("Weight (w1)")
    ax_path.set_ylabel("Weight (w2)")
    ax_path.legend()

    ax_surf = fig.add_subplot(235, projection="3d")
    ax_surf.plot_surface(w1_grid, w2_grid, z_grid, cmap="viridis", alpha=0.6, edgecolor="none")
    path_line_3d, = ax_surf.plot([], [], [], color="black", marker="o", markersize=3,
                                 linewidth=2, label="Optimizer Path")
    ax_surf.plot([w1_opt], [w2_opt], [min_cost], marker="*", color="red", markersize=12,
                 label="Global Min")
    ax_surf.set_title("4. 3D Gradient Path")
    ax_surf.set_xlabel("Weight (w1)")
    ax_surf.set_ylabel("Weight (w2)")
    ax_surf.set_zlabel("Cost (MSE)")
    ax_surf.view_init(elev=30, azim=-60)

    def update(frame: int):
        """Advance every panel to epoch ``frame`` of the recorded history."""
        nonlocal fit_plane
        w1, w2, b = history.w1[frame], history.w2[frame], history.bias[frame]

        fit_plane.remove()
        fit_plane = ax_fit.plot_surface(fit_x1, fit_x2, w1 * fit_x1 + w2 * fit_x2 + b,
                                        color="red", alpha=0.3)

        cost_line.set_data(range(frame + 1), history.cost[:frame + 1])
        costw_line.set_data(history.w1[:frame + 1], history.cost[:frame + 1])
        path_line.set_data(history.w1[:frame + 1], history.w2[:frame + 1])
        path_line_3d.set_data(history.w1[:frame + 1], history.w2[:frame + 1])
        path_line_3d.set_3d_properties(history.cost[:frame + 1])

        ax_fit.set_title(f"1. Fit (Epoch {frame}): y = {w1:.2f}x1 + {w2:.2f}x2 + {b:.2f}")
        ax_curve.set_title(f"2. Learning Curve: Cost = {history.cost[frame]:.4f}")
        return cost_line, costw_line, path_line, path_line_3d

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=len(history.cost),
                         interval=FRAME_INTERVAL, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    """Generate the dataset, train while recording history, then show the animation."""
    x, y    = load_dataset()
    history = train_and_record(x, y)
    animate(x, y, history)


if __name__ == "__main__":
    main()
