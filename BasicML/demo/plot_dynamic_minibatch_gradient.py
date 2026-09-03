# AI generated (refactored/authored with Claude Code)
"""Animated mini-batch vs full-batch (BatchGD) gradient directions.

Run directly: python BasicML/demo/plot_dynamic_minibatch_gradient.py

On a synthetic ``y = w*x + b`` dataset this demo makes the mini-batch gradient
*noise* visible. At every epoch it freezes the current ``(w, b)`` and computes,
at that one point:

  * the full-batch gradient over all samples (BatchGD), and
  * the gradient of each individual mini-batch.

Panel 1 draws them as descent directions (arrows along ``-gradient``): one bold
arrow for the full batch, a thin fan for the mini-batches. The spread of that
fan is exactly why a raw mini-batch SGD step wobbles from update to update --
and why Momentum, which keeps a running average of successive directions,
smooths the path. Two trajectories are traced on the cost contour: mini-batch
SGD and mini-batch Momentum, from the same start.

Panels: (1) cost contour with the gradient fan + both optimizer paths,
(2) current fit, (3) full-batch learning curves, (4) per-batch angle between the
mini-batch descent direction and the full-batch one.
"""
import os
import sys
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.datasets        import iter_minibatches
from basicml.nn.linear       import Linear
from basicml.nn.loss         import MSELoss
from basicml.optim.sgd       import SGD
from basicml.optim.momentum  import Momentum

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED        = 0
N_SAMPLES   = 256
TRUE_W      = 2.5
TRUE_B      = 4.0
NOISE_STD   = 2.5
X_SPREAD    = 3.0

BATCH_SIZE  = 32                       # -> 8 mini-batches per epoch
INIT_W      = -2.0
INIT_B      = -3.0

EPOCHS      = 80
SGD_LR      = 0.03
MOMENTUM    = 0.9
# Momentum accumulates ~1/(1-MOMENTUM) of a step, so scale its lr down by that
# factor for a fair comparison -- otherwise it just takes 10x larger steps.
MOM_LR      = SGD_LR * (1.0 - MOMENTUM)

GRID_RESOLUTION = 60
ARROW_FRAC      = 0.11                 # arrow length as a fraction of the w-range
FRAME_INTERVAL  = 90                   # ms between frames
FIG_SIZE        = (15, 10)
# ----------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Per-epoch trajectory for both optimizers plus the gradient fan.

    Attributes:
        w_sgd, b_sgd, cost_sgd: mini-batch SGD parameter path and full-batch MSE.
        w_mom, b_mom, cost_mom: mini-batch Momentum path and full-batch MSE.
        full_dir: Unit descent direction ``-g/|g|`` of the full-batch gradient,
            evaluated at the SGD path point, shape ``(EPOCHS, 2)``.
        batch_dirs: One ``(n_batches, 2)`` array of unit descent directions per
            epoch, each row a mini-batch gradient at the same SGD path point.
        batch_angles: One ``(n_batches,)`` array of angles in degrees per epoch
            between each mini-batch descent direction and ``full_dir``.
    """
    w_sgd:   list[float]      = field(default_factory=list)
    b_sgd:   list[float]      = field(default_factory=list)
    cost_sgd: list[float]     = field(default_factory=list)
    w_mom:   list[float]      = field(default_factory=list)
    b_mom:   list[float]      = field(default_factory=list)
    cost_mom: list[float]     = field(default_factory=list)
    full_dir: list[np.ndarray]   = field(default_factory=list)
    batch_dirs: list[np.ndarray] = field(default_factory=list)
    batch_angles: list[np.ndarray] = field(default_factory=list)


def make_linear_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Sample a noisy 1-D linear dataset ``y = TRUE_W * x + TRUE_B + eps``.

    Returns:
        Tuple ``(x, y)`` of float64 arrays, each shaped ``(N_SAMPLES, 1)``.
    """
    rng = np.random.RandomState(SEED)
    x   = rng.uniform(-X_SPREAD, X_SPREAD, size=(N_SAMPLES, 1))
    y   = TRUE_W * x + TRUE_B + rng.normal(0.0, NOISE_STD, size=(N_SAMPLES, 1))
    return x, y


def mse_gradient(w: float, b: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the mean-squared-error cost w.r.t. ``(w, b)`` at one point.

    Matches :class:`basicml.nn.loss.MSELoss` (cost ``1/(2n) Σ (ŷ - y)²``), so
    ``dJ/dw = mean(x·resid)`` and ``dJ/db = mean(resid)``.

    Args:
        w: Weight value to evaluate the gradient at.
        b: Bias value to evaluate the gradient at.
        x: Input features, shape ``(n, 1)``.
        y: Targets, shape ``(n, 1)``.

    Returns:
        Array ``[dJ/dw, dJ/db]`` of shape ``(2,)``.
    """
    resid = (x * w + b) - y
    return np.array([float(np.mean(x * resid)), float(np.mean(resid))])


def unit_descent(gradient: np.ndarray) -> np.ndarray:
    """Return the unit-length descent direction ``-g / |g|`` (zero-safe)."""
    norm = float(np.linalg.norm(gradient))
    if norm == 0.0:
        return np.zeros(2)
    return -gradient / norm


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two 2-D vectors (0 if either is zero)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    cos = np.clip(float(np.dot(u, v) / (nu * nv)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _fresh_model() -> Linear:
    model = Linear(in_features=1, out_features=1)
    assert model.b is not None
    model.w.data = np.array([[INIT_W]], dtype=np.float64)
    model.b.data = np.array([[INIT_B]], dtype=np.float64)
    return model


def train_and_record(x: np.ndarray, y: np.ndarray) -> TrainingHistory:
    """Run mini-batch SGD and Momentum in lock-step, recording the fan each epoch.

    Both optimizers see the identical shuffled batch schedule every epoch. The
    gradient fan (full batch vs per batch) is measured at the *SGD* path point
    before that epoch's steps are taken, so the arrows show where each batch
    would pull from the current position.

    Args:
        x: Input features, shape ``(n, 1)``.
        y: Targets, shape ``(n, 1)``.

    Returns:
        A populated :class:`TrainingHistory`.
    """
    sgd_model, mom_model = _fresh_model(), _fresh_model()
    assert sgd_model.b is not None and mom_model.b is not None
    sgd_loss,  mom_loss  = MSELoss(), MSELoss()
    sgd_opt = SGD(sgd_model.parameters(), lr=SGD_LR)
    mom_opt = Momentum(mom_model.parameters(), lr=MOM_LR, momentum=MOMENTUM)

    hist = TrainingHistory()
    print("Training (mini-batch SGD vs Momentum) to gather history...")
    for epoch in range(EPOCHS):
        w_s, b_s = float(sgd_model.w.data[0, 0]), float(sgd_model.b.data[0, 0])
        w_m, b_m = float(mom_model.w.data[0, 0]), float(mom_model.b.data[0, 0])

        # Freeze the SGD point, then measure every batch gradient there.
        batches   = list(iter_minibatches(x, y, BATCH_SIZE, random_state=SEED + epoch))
        full_g    = mse_gradient(w_s, b_s, x, y)
        batch_gs  = [mse_gradient(w_s, b_s, xb, yb) for xb, yb in batches]
        full_dir  = unit_descent(full_g)

        hist.w_sgd.append(w_s); hist.b_sgd.append(b_s)
        hist.w_mom.append(w_m); hist.b_mom.append(b_m)
        hist.cost_sgd.append(sgd_loss(sgd_model(x), y))
        hist.cost_mom.append(mom_loss(mom_model(x), y))
        hist.full_dir.append(full_dir)
        hist.batch_dirs.append(np.array([unit_descent(g) for g in batch_gs]))
        # Angle between gradients == angle between their descent directions.
        hist.batch_angles.append(
            np.array([angle_between(g, full_g) for g in batch_gs]))

        for xb, yb in batches:
            sgd_loss(sgd_model(xb), yb)
            sgd_model.backward(sgd_loss.backward())
            sgd_opt.step(); sgd_opt.zero_grad()

            mom_loss(mom_model(xb), yb)
            mom_model.backward(mom_loss.backward())
            mom_opt.step(); mom_opt.zero_grad()

    print(f"Done. Final full-batch MSE  ->  SGD {hist.cost_sgd[-1]:.4f} | "
          f"Momentum {hist.cost_mom[-1]:.4f}")
    return hist


def closed_form_optimum(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares optimum ``(w, b, mse)`` for simple linear regression."""
    w = float(np.cov(x.squeeze(), y.squeeze())[0, 1] / np.var(x))
    b = float(y.mean() - w * x.mean())
    cost = float(np.mean((w * x + b - y) ** 2) / 2.0)
    return w, b, cost


def cost_surface(x: np.ndarray, y: np.ndarray,
                 w_range: tuple[float, float],
                 b_range: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Evaluate the MSE cost ``1/(2n) Σ (ŷ - y)²`` on a ``(w, b)`` grid.

    Args:
        x: Input features, shape ``(n, 1)``.
        y: Targets, shape ``(n, 1)``.
        w_range: ``(min, max)`` weight range for the grid.
        b_range: ``(min, max)`` bias range for the grid.

    Returns:
        Tuple ``(w_grid, b_grid, z_grid)``, each ``(GRID_RESOLUTION, GRID_RESOLUTION)``.
    """
    w_vals = np.linspace(*w_range, GRID_RESOLUTION)
    b_vals = np.linspace(*b_range, GRID_RESOLUTION)
    w_grid, b_grid = np.meshgrid(w_vals, b_vals)

    preds  = w_grid[..., None, None] * x + b_grid[..., None, None]
    z_grid = np.mean((preds - y) ** 2, axis=(-2, -1)) / 2.0
    return w_grid, b_grid, z_grid


def animate(x: np.ndarray, y: np.ndarray, history: TrainingHistory) -> FuncAnimation:
    """Build the 4-panel figure and animate it over the recorded history.

    Args:
        x: Input features, shape ``(n, 1)``.
        y: Targets, shape ``(n, 1)``.
        history: Recorded trajectory from :func:`train_and_record`.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation` handle.
    """
    w_sgd = np.array(history.w_sgd); b_sgd = np.array(history.b_sgd)
    w_mom = np.array(history.w_mom); b_mom = np.array(history.b_mom)
    cost_sgd = np.array(history.cost_sgd); cost_mom = np.array(history.cost_mom)
    n_batches = len(history.batch_dirs[0])

    w_opt, b_opt, min_cost = closed_form_optimum(x, y)

    all_w = np.concatenate([w_sgd, w_mom, [w_opt]])
    all_b = np.concatenate([b_sgd, b_mom, [b_opt]])
    w_margin = max(float(np.ptp(all_w)), 1.0) * 0.35
    b_margin = max(float(np.ptp(all_b)), 1.0) * 0.35
    w_range  = (all_w.min() - w_margin, all_w.max() + w_margin)
    b_range  = (all_b.min() - b_margin, all_b.max() + b_margin)
    arrow_len = (w_range[1] - w_range[0]) * ARROW_FRAC

    w_grid, b_grid, z_grid = cost_surface(x, y, w_range, b_range)

    fig = plt.figure(figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            "BasicML - Mini-batch vs Full-batch Gradient Directions")

    # --- Panel 1: cost contour + gradient fan + both optimizer paths --------
    ax_path = fig.add_subplot(221)
    contour = ax_path.contour(w_grid, b_grid, z_grid,
                              levels=np.linspace(min_cost, z_grid.max(), 18),
                              cmap="viridis", alpha=0.7)
    ax_path.clabel(contour, inline=True, fontsize=7)
    ax_path.plot([w_opt], [b_opt], marker="*", color="red", markersize=14,
                 label=f"Least-squares min ({w_opt:.2f}, {b_opt:.2f})")
    sgd_path, = ax_path.plot([], [], color="black", marker="o", markersize=3,
                             linewidth=1, alpha=0.75, label="mini-batch SGD path")
    mom_path, = ax_path.plot([], [], color="tab:orange", marker="o", markersize=3,
                             linewidth=1, alpha=0.9, label="mini-batch Momentum path")
    # Thin fan: per-mini-batch descent directions from the current SGD point.
    fan_quiver = ax_path.quiver(
        np.zeros(n_batches), np.zeros(n_batches),
        np.zeros(n_batches), np.zeros(n_batches),
        angles="xy", scale_units="xy", scale=1.0,
        color="tab:blue", alpha=0.55, width=0.004, label="per mini-batch grad")
    # Bold arrow: full-batch (BatchGD) descent direction.
    full_quiver = ax_path.quiver(
        [0.0], [0.0], [0.0], [0.0],
        angles="xy", scale_units="xy", scale=1.0,
        color="crimson", width=0.011, label="full-batch (BatchGD) grad")
    ax_path.set_xlim(*w_range); ax_path.set_ylim(*b_range)
    ax_path.set_title("1. Gradient fan on the MSE cost surface")
    ax_path.set_xlabel("Weight (w)"); ax_path.set_ylabel("Bias (b)")
    ax_path.legend(fontsize=7, loc="lower right")

    # --- Panel 2: current fit --------------------------------------------------
    ax_fit = fig.add_subplot(222)
    ax_fit.scatter(x, y, color="tab:blue", alpha=0.35, s=15, label="data")
    x_line = np.array([x.min() - 1, x.max() + 1])
    fit_sgd, = ax_fit.plot([], [], color="black", linewidth=2, label="SGD fit")
    fit_mom, = ax_fit.plot([], [], color="tab:orange", linewidth=2, label="Momentum fit")
    ax_fit.plot(x_line, TRUE_W * x_line + TRUE_B, color="red", linestyle="--",
                linewidth=1, alpha=0.7, label="ground truth")
    ax_fit.set_xlim(*x_line); ax_fit.set_ylim(y.min() - 2, y.max() + 2)
    ax_fit.set_title("2. Current fit")
    ax_fit.set_xlabel("X"); ax_fit.set_ylabel("Y")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 3: full-batch learning curves --------------------------------
    ax_curve = fig.add_subplot(223)
    curve_sgd, = ax_curve.plot([], [], color="black", linewidth=2, label="SGD")
    curve_mom, = ax_curve.plot([], [], color="tab:orange", linewidth=2, label="Momentum")
    dot_sgd, = ax_curve.plot([], [], "o", color="black", markersize=6)
    dot_mom, = ax_curve.plot([], [], "o", color="tab:orange", markersize=6)
    ax_curve.set_xlim(0, EPOCHS)
    ax_curve.set_ylim(0, max(cost_sgd.max(), cost_mom.max()) * 1.1)
    ax_curve.set_title("3. Full-batch learning curve")
    ax_curve.set_xlabel("Epoch"); ax_curve.set_ylabel("Cost (MSE / 2)")
    ax_curve.legend(fontsize=8)
    ax_curve.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 4: per-batch angle to the full-batch direction --------------
    ax_ang = fig.add_subplot(224)
    bars = ax_ang.bar(np.arange(n_batches), np.zeros(n_batches),
                      color="tab:blue", alpha=0.75)
    ax_ang.axhline(90, color="gray", linestyle=":", linewidth=1)
    ax_ang.set_xlim(-0.6, n_batches - 0.4)
    ax_ang.set_ylim(0, 180)
    ax_ang.set_title("4. Angle of each mini-batch gradient vs full-batch")
    ax_ang.set_xlabel("mini-batch index"); ax_ang.set_ylabel("angle (degrees)")
    ax_ang.grid(True, axis="y", linestyle="--", alpha=0.5)

    def update(frame: int):
        """Advance every panel to epoch ``frame`` of the recorded history."""
        w_s, b_s = w_sgd[frame], b_sgd[frame]
        w_m, b_m = w_mom[frame], b_mom[frame]

        sgd_path.set_data(w_sgd[:frame + 1], b_sgd[:frame + 1])
        mom_path.set_data(w_mom[:frame + 1], b_mom[:frame + 1])

        batch_dirs = history.batch_dirs[frame] * arrow_len
        fan_quiver.set_offsets(np.tile([w_s, b_s], (n_batches, 1)))
        fan_quiver.set_UVC(batch_dirs[:, 0], batch_dirs[:, 1])
        full_dir = history.full_dir[frame] * arrow_len
        full_quiver.set_offsets(np.array([[w_s, b_s]]))
        full_quiver.set_UVC([full_dir[0]], [full_dir[1]])

        fit_sgd.set_data(x_line, w_s * x_line + b_s)
        fit_mom.set_data(x_line, w_m * x_line + b_m)

        epochs = np.arange(frame + 1)
        curve_sgd.set_data(epochs, cost_sgd[:frame + 1])
        curve_mom.set_data(epochs, cost_mom[:frame + 1])
        dot_sgd.set_data([frame], [cost_sgd[frame]])
        dot_mom.set_data([frame], [cost_mom[frame]])

        angles = history.batch_angles[frame]
        for rect, a in zip(bars, angles):
            rect.set_height(a)

        ax_path.set_title(
            f"1. Gradient fan (epoch {frame}) -- fan spread "
            f"{angles.max() - angles.min():.0f} deg")
        ax_ang.set_title(
            f"4. Mini-batch angle vs full-batch -- mean {angles.mean():.0f} deg, "
            f"max {angles.max():.0f} deg")
        return (sgd_path, mom_path, fan_quiver, full_quiver,
                fit_sgd, fit_mom, curve_sgd, curve_mom, dot_sgd, dot_mom, *bars)

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=EPOCHS,
                         interval=FRAME_INTERVAL, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    """Sample the dataset, train while recording history, then show the animation."""
    x, y    = make_linear_dataset()
    history = train_and_record(x, y)
    animate(x, y, history)


if __name__ == "__main__":
    main()
