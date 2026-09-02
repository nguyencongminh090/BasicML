# AI generated (refactored/authored with Claude Code)
"""Animated 3D cost-surface cho logistic regression tren du lieu 1D.

Chay truc tiep: python BasicML/demo/plot_dynamic_3d_logistic.py
Chi 1 panel: mat cost BCE trong khong gian (w, b) va duong di momentum.
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
from basicml.optim.momentum import Momentum

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
MOM_CYCLE      = (0.95, 0.98, 0.95)   # (peak, base, final)
WARMUP_FRAC    = 0.3

GRID_RESOLUTION = 50
GRID_MARGIN     = 7.5                  # do rong (w, b) quanh quy dao
Z_HEADROOM      = 15.0                 # chan tren cua truc cost
FRAME_INTERVAL  = 10
FIG_SIZE        = (10, 8)
# ----------------------------------------------------------------------


@dataclass
class OneCycle:
    """Lich trinh one-cycle: cosine warmup roi cosine anneal."""
    lr:          tuple[float, float, float]
    momentum:    tuple[float, float, float]
    warmup_frac: float

    def at(self, progress: float) -> tuple[float, float]:
        lr_base, lr_peak, lr_final    = self.lr
        mom_peak, mom_base, mom_final = self.momentum

        if progress < self.warmup_frac:
            phase  = progress / self.warmup_frac
            factor = 0.5 * (1 - np.cos(np.pi * phase))
            lr     = lr_base + (lr_peak - lr_base) * factor
            mom    = mom_peak - (mom_peak - mom_base) * factor
        else:
            phase  = (progress - self.warmup_frac) / (1.0 - self.warmup_frac)
            factor = 0.5 * (1 + np.cos(np.pi * phase))
            lr     = lr_final + (lr_peak - lr_final) * factor
            mom    = mom_final + (mom_base - mom_final) * factor
        return lr, mom


@dataclass
class TrainingHistory:
    weight: np.ndarray
    bias:   np.ndarray
    cost:   np.ndarray


def make_threshold_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng       = random.Random(SEED)
    lo, hi    = X_RANGE
    threshold = lo + LABEL_FRACTION * (hi - lo)

    x = np.array([rng.uniform(lo, hi) for _ in range(N_SAMPLES)]).reshape(-1, 1)
    y = (x > threshold).astype(np.float64)
    x = (x - x.mean()) / x.std()
    return x, y


def train_and_record(x: np.ndarray, y: np.ndarray) -> TrainingHistory:
    linear = Linear(in_features=1, out_features=1)
    assert linear.b is not None
    linear.w.data = np.array([[INIT_W]])
    linear.b.data = np.array([[INIT_B]])

    model     = Sequential(linear, Sigmoid())
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LR_CYCLE[0], momentum=MOM_CYCLE[0])
    schedule  = OneCycle(LR_CYCLE, MOM_CYCLE, WARMUP_FRAC)

    weight_hist: list[float] = []
    bias_hist:   list[float] = []
    cost_hist:   list[float] = []

    print("Training model to gather history...")
    for epoch in range(EPOCHS):
        optimizer.lr, optimizer.momentum = schedule.at(epoch / EPOCHS)

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
    w_vals = np.linspace(*w_range, GRID_RESOLUTION)
    b_vals = np.linspace(*b_range, GRID_RESOLUTION)
    w_grid, b_grid = np.meshgrid(w_vals, b_vals)

    logits = w_grid[..., None, None] * x + b_grid[..., None, None]
    probs  = np.clip(1 / (1 + np.exp(-logits)), 1e-15, 1 - 1e-15)
    z_grid = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs), axis=(-2, -1))
    return w_grid, b_grid, z_grid


def animate(x: np.ndarray, y: np.ndarray, history: TrainingHistory) -> FuncAnimation:
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
                            linewidth=2, label="Momentum Path")
    ax.plot([w_opt], [b_opt], [min_cost], marker="*", color="red", markersize=12, label="End")
    ax.set_box_aspect((2, 2, 1))
    ax.set_title("3D Gradient Path on Cost Surface (Logistic Regression)")
    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Bias (b)")
    ax.set_zlabel("Cost (BCE)")
    ax.view_init(elev=30, azim=-60)
    ax.legend()

    def update(frame: int):
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
    x, y    = make_threshold_dataset()
    history = train_and_record(x, y)
    animate(x, y, history)


if __name__ == "__main__":
    main()
