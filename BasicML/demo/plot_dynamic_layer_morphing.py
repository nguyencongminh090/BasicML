# AI generated (refactored/authored with Claude Code)
"""Animated: khong gian dac trung bi "bien hinh" qua tung lop cua MLP sau.

Chay truc tiep: python BasicML/demo/plot_dynamic_layer_morphing.py
3 panel: bien quyet dinh o khong gian dau vao, diem du lieu morphing qua
tung lop, va luoi toa do bi bien dang qua tung lop. Khong gian > 2D duoc
chieu ve 2D bang PCA.
"""
import os
import sys
from typing import Callable, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.datasets       import make_moons
from basicml.nn.sequential  import Sequential
from basicml.nn.linear      import Linear
from basicml.nn.activation  import Sigmoid
from basicml.nn.module      import Module
from basicml.nn.loss        import BinaryCrossEntropy
from basicml.optim.momentum import Momentum
from basicml.visualize      import plot_decision_boundary

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED           = 42
N_SAMPLES      = 200
NOISE          = 0.15

HIDDEN_DIMS    = [16, 12, 8]           # kich thuoc cac lop an; doi tu do
ACTIVATION: Callable[[], Module] = Sigmoid
INIT_TYPE      = "xavier"
LEARN_RATE     = 0.20
MOMENTUM       = 0.92
EPOCHS         = 5000

GRID_LINES     = 22
GRID_POINTS    = 120
X_LINE_SPAN    = (-1.5, 2.5)
Y_LINE_SPAN    = (-1.0, 1.5)

TRANSITION_FRAMES = 45                 # frame cho moi lan morph giua 2 lop
PAUSE_FRAMES      = 15                 # frame dung lai o moi lop
RESET_FRAMES      = 60                 # frame morph tu lop cuoi ve dau vao
FRAME_INTERVAL    = 35
FIG_SIZE          = (20, 6.5)
# ----------------------------------------------------------------------


def build_model() -> Sequential:
    layers: list[Module] = []
    in_dim = 2
    for out_dim in HIDDEN_DIMS:
        layers.append(Linear(in_dim, out_dim, init_type=INIT_TYPE))
        layers.append(ACTIVATION())
        in_dim = out_dim
    layers.append(Linear(in_dim, 1, init_type=INIT_TYPE))
    layers.append(ACTIVATION())
    return Sequential(*layers)


def train(model: Sequential, x: np.ndarray, y: np.ndarray) -> None:
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    y_pred = model(x)
    for _ in range(EPOCHS):
        y_pred = model(x)
        criterion(y_pred, y)  # cap nhat cache cho backward()
        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

    accuracy = float(np.mean((y_pred >= 0.5).astype(int) == y)) * 100.0
    print(f"Network trained ({len(model.layers)} layers). Accuracy: {accuracy:.1f}%")


def project_to_2d(data: np.ndarray,
                  basis: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Chieu ve (N, 2). Voi dau ra 1D thi dat truc Y = 0."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[1] == 1:
        return np.column_stack([data.ravel(), np.zeros(len(data))]), None
    if data.shape[1] == 2:
        return data, np.eye(2)

    if basis is None:
        centered = data - data.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis    = vt[:2].T
    return data @ basis, basis


def project_lines(lines: list[np.ndarray],
                  basis: Optional[np.ndarray]) -> list[np.ndarray]:
    """Ep moi duong luoi ve dang (N, 2)."""
    out = []
    for line in lines:
        line = np.asarray(line)
        if line.ndim == 1:
            line = line.reshape(-1, 1)
        if line.shape[1] == 1:
            out.append(np.column_stack([line.ravel(), np.zeros(len(line))]))
        elif line.shape[1] == 2:
            out.append(line)
        elif basis is not None:
            out.append(line @ basis)
        else:
            out.append(line[:, :2])
    return out


def make_grid_lines() -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_lo, x_hi = X_LINE_SPAN
    y_lo, y_hi = Y_LINE_SPAN
    horizontal = [np.column_stack([np.linspace(x_lo, x_hi, GRID_POINTS),
                                   np.full(GRID_POINTS, y_val)])
                  for y_val in np.linspace(y_lo, y_hi, GRID_LINES)]
    vertical   = [np.column_stack([np.full(GRID_POINTS, x_val),
                                   np.linspace(y_lo, y_hi, GRID_POINTS)])
                  for x_val in np.linspace(x_lo, x_hi, GRID_LINES)]
    return horizontal, vertical


def collect_stages(model: Sequential, x: np.ndarray,
                   horizontal: list[np.ndarray], vertical: list[np.ndarray]):
    """Bieu dien 2D cua diem + luoi sau tung lop (stage 0 = dau vao)."""
    stage_points     = [x]
    stage_horizontal = [horizontal]
    stage_vertical   = [vertical]
    stage_names      = ["Input Space $X$ (2D)"]

    cur_points, cur_h, cur_v = x, horizontal, vertical
    for index, layer in enumerate(model.layers):
        cur_points = layer.forward(cur_points)
        cur_h      = [layer.forward(line) for line in cur_h]
        cur_v      = [layer.forward(line) for line in cur_v]

        points_2d, basis = project_to_2d(cur_points)
        stage_points.append(points_2d)
        stage_horizontal.append(project_lines(cur_h, basis))
        stage_vertical.append(project_lines(cur_v, basis))
        stage_names.append(f"Layer {index + 1}: {layer.__class__.__name__} "
                           f"({cur_points.shape[1]}D)")

    return stage_points, stage_horizontal, stage_vertical, stage_names


def cosine_ease(t: float) -> float:
    return 0.5 * (1 - np.cos(np.pi * t))


def lerp(a, b, t: float):
    return (1 - t) * a + t * b


def animate(model: Sequential, x, y, stages) -> FuncAnimation:
    stage_points, stage_horizontal, stage_vertical, stage_names = stages
    num_stages   = len(stage_points)
    n_horizontal = len(stage_horizontal[0])
    n_vertical   = len(stage_vertical[0])
    labels       = y.ravel()

    cycle_len   = (TRANSITION_FRAMES + PAUSE_FRAMES) * (num_stages - 1)
    total_frames = cycle_len + RESET_FRAMES

    def state_at(frame: int):
        pos = frame % total_frames
        if pos >= cycle_len:
            t     = cosine_ease((pos - cycle_len) / RESET_FRAMES)
            src, dst = num_stages - 1, 0
            note  = f"Resetting to Input Space (t={t:.2f})"
        else:
            stage       = pos // (TRANSITION_FRAMES + PAUSE_FRAMES)
            within      = pos % (TRANSITION_FRAMES + PAUSE_FRAMES)
            src         = stage
            dst         = min(stage + 1, num_stages - 1)
            if within < PAUSE_FRAMES:
                t, note = 0.0, f"Active: {stage_names[src]}"
            else:
                t    = cosine_ease((within - PAUSE_FRAMES) / TRANSITION_FRAMES)
                note = f"{stage_names[src]} -> {stage_names[dst]} (t={t:.2f})"

        points = lerp(stage_points[src], stage_points[dst], t)
        horiz  = [lerp(stage_horizontal[src][j], stage_horizontal[dst][j], t)
                  for j in range(n_horizontal)]
        vert   = [lerp(stage_vertical[src][j], stage_vertical[dst][j], t)
                  for j in range(n_vertical)]
        return points, horiz, vert, note

    fig, (ax_boundary, ax_points, ax_grid) = plt.subplots(1, 3, figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"BasicML - Deep Network ({num_stages - 1} Layers) Space Morphing")

    plot_decision_boundary(pred_func=model, X=x, y=y, ax=ax_boundary, cmap="Spectral",
                           title="1. Decision Boundary (Input Space)", show=False)
    ax_boundary.set_xlabel("Feature $x_1$")
    ax_boundary.set_ylabel("Feature $x_2$")

    scatter = ax_points.scatter(x[:, 0], x[:, 1], c=labels, cmap="Spectral",
                                edgecolors="k", s=45, alpha=0.9)
    horiz_artists = [ax_grid.plot([], [], color="steelblue", alpha=0.6, linewidth=1.0)[0]
                     for _ in range(n_horizontal)]
    vert_artists  = [ax_grid.plot([], [], color="coral", alpha=0.6, linewidth=1.0)[0]
                     for _ in range(n_vertical)]

    def update(frame: int):
        points, horiz, vert, note = state_at(frame)
        scatter.set_offsets(points)
        for artist, line in zip(horiz_artists, horiz):
            artist.set_data(line[:, 0], line[:, 1])
        for artist, line in zip(vert_artists, vert):
            artist.set_data(line[:, 0], line[:, 1])

        min_x, max_x = min(points[:, 0].min(), -1.2) - 0.3, max(points[:, 0].max(), 1.2) + 0.3
        min_y, max_y = min(points[:, 1].min(), -1.2) - 0.3, max(points[:, 1].max(), 1.2) + 0.3
        for ax, title in ((ax_points, "2. Data Points Morphing across Layers"),
                          (ax_grid,   "3. Manifold Deformation across Layers")):
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_title(f"{title}\n{note}", fontsize=10)
            ax.set_xlabel("Latent Dim 1 (PCA Projected)")
            ax.set_ylabel("Latent Dim 2 (PCA Projected)")
        ax_grid.grid(True, linestyle="--", alpha=0.4)
        return scatter, *horiz_artists, *vert_artists

    print("Rendering deep multi-layer morphing animation...")
    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=FRAME_INTERVAL, blit=False, repeat=True)
    plt.suptitle("BasicML - Deep Neural Network Layer-by-Layer Space Morphing",
                 fontsize=13, y=0.98)
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    np.random.seed(SEED)
    x, y  = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    model = build_model()
    train(model, x, y)

    horizontal, vertical = make_grid_lines()
    stages = collect_stages(model, x, horizontal, vertical)
    animate(model, x, y, stages)


if __name__ == "__main__":
    main()
