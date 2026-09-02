# AI generated (refactored/authored with Claude Code)
"""Animated decision boundary cua mot MLP nho tren dataset moons.

Chay truc tiep: python BasicML/demo/plot_dynamic_decision_boundary.py
Panel trai: vung quyet dinh theo epoch. Panel phai: learning curve.
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
from basicml.optim.momentum import Momentum

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED           = 45
N_SAMPLES      = 200
NOISE          = 0.20

HIDDEN_DIM     = 5
LEARN_RATE     = 0.08
MOMENTUM       = 0.92
EPOCHS         = 2000
RECORD_EVERY   = 6                      # luu lich su moi bao nhieu epoch

GRID_STEP      = 0.03                   # do min cua luoi ve bien quyet dinh
GRID_PADDING   = 0.5
FRAME_INTERVAL = 5
FIG_SIZE       = (14, 6)
# ----------------------------------------------------------------------


def build_model() -> Sequential:
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
    x_min, x_max = x[:, 0].min() - GRID_PADDING, x[:, 0].max() + GRID_PADDING
    y_min, y_max = x[:, 1].min() - GRID_PADDING, x[:, 1].max() + GRID_PADDING
    xx, yy = np.meshgrid(np.arange(x_min, x_max, GRID_STEP),
                         np.arange(y_min, y_max, GRID_STEP))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid_points


def train_and_record(model: Sequential, x: np.ndarray, y: np.ndarray):
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
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
    np.random.seed(SEED)
    x, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    model = build_model()
    animate(x, y, *train_and_record(model, x, y))


if __name__ == "__main__":
    main()
