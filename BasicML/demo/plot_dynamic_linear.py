# AI generated (refactored/authored with Claude Code)
"""Animated linear-regression training tren BasicML/data.csv.

Chay truc tiep: python BasicML/demo/plot_dynamic_linear.py
Hien 5 panel: duong fit, learning curve, cost-vs-w, va duong di gradient
(2D contour + 3D surface) voi lich trinh one-cycle cho lr/momentum.
"""
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.nn.linear      import Linear
from basicml.nn.loss        import MSELoss
from basicml.optim.momentum import Momentum

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(REPO_ROOT, "data.csv")
X_COLUMNS   = ["X"]
Y_COLUMNS   = ["Y"]

INIT_W      = -1.0
INIT_B      = -2.0

MAX_EPOCHS  = 1000
LR_CYCLE    = (0.005, 0.08, 0.001)    # (base, peak, final)
MOM_CYCLE   = (0.95, 0.70, 0.875)     # (peak, base, final) — theo one_cycle()
WARMUP_FRAC = 0.3

EARLY_STOP_PATIENCE  = 15
EARLY_STOP_MIN_DELTA = 1e-4

GRID_RESOLUTION = 50
FRAME_INTERVAL  = 5                    # ms giua cac frame
FIG_SIZE        = (18, 10)
# ----------------------------------------------------------------------


@dataclass
class OneCycle:
    """Lich trinh one-cycle: cosine warmup roi cosine anneal."""
    lr:          tuple[float, float, float]   # (base, peak, final)
    momentum:    tuple[float, float, float]   # (peak, base, final)
    warmup_frac: float

    def at(self, progress: float) -> tuple[float, float]:
        lr_base, lr_peak, lr_final    = self.lr
        mom_peak, mom_base, mom_final = self.momentum

        if progress < self.warmup_frac:
            phase  = progress / self.warmup_frac
            factor = 0.5 * (1 - np.cos(np.pi * phase))          # 0 -> 1
            lr     = lr_base + (lr_peak - lr_base) * factor
            mom    = mom_peak - (mom_peak - mom_base) * factor
        else:
            phase  = (progress - self.warmup_frac) / (1.0 - self.warmup_frac)
            factor = 0.5 * (1 + np.cos(np.pi * phase))          # 1 -> 0
            lr     = lr_final + (lr_peak - lr_final) * factor
            mom    = mom_final + (mom_base - mom_final) * factor
        return lr, mom


@dataclass
class TrainingHistory:
    weight: np.ndarray
    bias:   np.ndarray
    cost:   np.ndarray


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    x     = frame[X_COLUMNS].to_numpy(dtype=np.float64)
    y     = frame[Y_COLUMNS].to_numpy(dtype=np.float64)
    return x, y


def train_and_record(x: np.ndarray, y: np.ndarray) -> TrainingHistory:
    model = Linear(in_features=1, out_features=1)
    assert model.b is not None
    model.w.data = np.array([[INIT_W]])
    model.b.data = np.array([[INIT_B]])

    criterion = MSELoss()
    optimizer = Momentum(model.parameters(), lr=LR_CYCLE[0], momentum=MOM_CYCLE[0])
    schedule  = OneCycle(LR_CYCLE, MOM_CYCLE, WARMUP_FRAC)

    weight_hist: list[float] = []
    bias_hist:   list[float] = []
    cost_hist:   list[float] = []

    best_cost   = float("inf")
    no_improve  = 0

    print("Training model to gather history...")
    for epoch in range(MAX_EPOCHS):
        progress = epoch / MAX_EPOCHS
        optimizer.lr, optimizer.momentum = schedule.at(progress)

        weight_hist.append(model.w.data[0, 0])
        bias_hist.append(model.b.data[0, 0])

        cost = criterion(model(x), y)
        cost_hist.append(cost)

        if progress >= WARMUP_FRAC:
            if cost < best_cost - EARLY_STOP_MIN_DELTA:
                best_cost, no_improve = cost, 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stop at epoch {epoch + 1} "
                          f"(best cost {best_cost:.6f})")
                    break

        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

    print(f"Training complete. Final cost: {cost_hist[-1]:.4f}")
    return TrainingHistory(
        weight=np.array(weight_hist),
        bias=np.array(bias_hist),
        cost=np.array(cost_hist),
    )


def closed_form_optimum(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    w = float(np.cov(x.squeeze(), y.squeeze())[0, 1] / np.var(x))
    b = float(y.mean() - w * x.mean())
    cost = float(np.mean((w * x + b - y) ** 2))
    return w, b, cost


def cost_surface(x: np.ndarray, y: np.ndarray,
                 w_range: tuple[float, float],
                 b_range: tuple[float, float]) -> tuple[np.ndarray, ...]:
    w_vals = np.linspace(*w_range, GRID_RESOLUTION)
    b_vals = np.linspace(*b_range, GRID_RESOLUTION)
    w_grid, b_grid = np.meshgrid(w_vals, b_vals)

    preds = w_grid[..., None, None] * x + b_grid[..., None, None]
    z_grid = np.mean((preds - y) ** 2, axis=(-2, -1))
    return w_grid, b_grid, z_grid


def animate(x: np.ndarray, y: np.ndarray, history: TrainingHistory) -> FuncAnimation:
    w_opt, b_opt, min_cost = closed_form_optimum(x, y)

    w_margin = max(float(np.ptp(history.weight)), 2.0) * 0.4
    b_margin = max(float(np.ptp(history.bias)), 2.0) * 0.4
    w_range  = (min(history.weight.min(), w_opt) - w_margin,
                max(history.weight.max(), w_opt) + w_margin)
    b_range  = (min(history.bias.min(), b_opt) - b_margin,
                max(history.bias.max(), b_opt) + b_margin)

    w_grid, b_grid, z_grid = cost_surface(x, y, w_range, b_range)

    fig = plt.figure(figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Linear Regression Dynamic Training")

    ax_fit  = fig.add_subplot(231)
    ax_fit.scatter(x, y, color="blue", alpha=0.6, label="Training Data")
    fit_line, = ax_fit.plot([], [], color="red", linewidth=2, label="Fitted Line")
    ax_fit.set_xlim(x.min() - 1, x.max() + 1)
    ax_fit.set_ylim(y.min() - 2, y.max() + 2)
    ax_fit.set_title("1. Linear Regression Fit")
    ax_fit.set_xlabel("X")
    ax_fit.set_ylabel("Y")
    ax_fit.legend()
    ax_fit.grid(True, linestyle="--", alpha=0.6)

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
    costw_line, = ax_costw.plot([], [], color="purple", linewidth=2, label="Cost vs W")
    ax_costw.set_xlim(history.weight.min() - 0.5, history.weight.max() + 0.5)
    ax_costw.set_ylim(0, history.cost.max() * 1.1)
    ax_costw.set_title("5. Cost vs Weight (w)")
    ax_costw.set_xlabel("Weight (w)")
    ax_costw.set_ylabel("Cost (MSE)")
    ax_costw.legend()
    ax_costw.grid(True, linestyle="--", alpha=0.6)

    ax_path = fig.add_subplot(234)
    contour = ax_path.contour(w_grid, b_grid, z_grid,
                              levels=np.linspace(min_cost, z_grid.max(), 20),
                              cmap="viridis", alpha=0.8)
    ax_path.clabel(contour, inline=True, fontsize=8)
    ax_path.plot([w_opt], [b_opt], marker="*", color="red", markersize=12,
                 label=f"Global Min ({w_opt:.2f}, {b_opt:.2f})")
    path_line, = ax_path.plot([], [], color="black", marker="o", markersize=3,
                              linewidth=1, alpha=0.7, label="Momentum Path")
    ax_path.set_xlim(*w_range)
    ax_path.set_ylim(*b_range)
    ax_path.set_title("3. 2D Gradient Path on Cost Surface")
    ax_path.set_xlabel("Weight (w)")
    ax_path.set_ylabel("Bias (b)")
    ax_path.legend()

    ax_surf = fig.add_subplot(235, projection="3d")
    ax_surf.plot_surface(w_grid, b_grid, z_grid, cmap="viridis", alpha=0.6, edgecolor="none")
    path_line_3d, = ax_surf.plot([], [], [], color="black", marker="o", markersize=3,
                                 linewidth=2, label="Momentum Path")
    ax_surf.plot([w_opt], [b_opt], [min_cost], marker="*", color="red", markersize=12,
                 label="Global Min")
    ax_surf.set_title("4. 3D Gradient Path")
    ax_surf.set_xlabel("Weight (w)")
    ax_surf.set_ylabel("Bias (b)")
    ax_surf.set_zlabel("Cost (MSE)")
    ax_surf.view_init(elev=30, azim=-60)

    def update(frame: int):
        w, b = history.weight[frame], history.bias[frame]
        fit_line.set_data(x.ravel(), (w * x + b).ravel())

        cost_line.set_data(range(frame + 1), history.cost[:frame + 1])
        costw_line.set_data(history.weight[:frame + 1], history.cost[:frame + 1])
        path_line.set_data(history.weight[:frame + 1], history.bias[:frame + 1])
        path_line_3d.set_data(history.weight[:frame + 1], history.bias[:frame + 1])
        path_line_3d.set_3d_properties(history.cost[:frame + 1])

        ax_fit.set_title(f"1. Fit (Epoch {frame}): y = {w:.2f}x + {b:.2f}")
        ax_curve.set_title(f"2. Learning Curve: Cost = {history.cost[frame]:.4f}")
        return fit_line, cost_line, costw_line, path_line, path_line_3d

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=len(history.cost),
                         interval=FRAME_INTERVAL, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    x, y    = load_dataset(DATA_PATH)
    history = train_and_record(x, y)
    animate(x, y, history)


if __name__ == "__main__":
    main()
