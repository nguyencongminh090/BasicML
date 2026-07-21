# AI generated

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(suppress=True, precision=4)

from basicml.nn.linear import LinearRegression
from basicml.nn.loss import MSELoss
from basicml.optim.momentum import Momentum


def main():
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data.csv')
    data      = pd.read_csv(data_path)

    X = data[['X']].values
    Y = data[['Y']].values

    x_mean = X.mean()
    x_std  = X.std()
    X      = (X - x_mean) / x_std

    epochs = 1000
    model  = LinearRegression(features=1)

    model.w.data = np.array([[-1.0]])
    model.b.data = np.array([[-2.0]])

    loss = MSELoss()

    base_lr, max_lr, final_lr    = 0.005, 0.08, 0.001
    max_mom, base_mom, final_mom = 0.95, 0.70, 0.875
    pct_start                    = 0.3

    optim      = Momentum(model.parameters(), lr=base_lr, momentum=max_mom)
    patience   = 15
    min_delta  = 1e-4
    best_loss  = float('inf')
    no_improve = 0

    w_hist    = []
    b_hist    = []
    loss_hist = []

    print("Training model to gather history...")
    for epoch in range(epochs):
        pct = epoch / epochs
        if pct < pct_start:
            phase_pct      = pct / pct_start
            factor         = 0.5 * (1 - np.cos(np.pi * phase_pct))
            optim.lr       = base_lr + (max_lr - base_lr) * factor
            optim.momentum = max_mom - (max_mom - base_mom) * factor
        else:
            phase_pct      = (pct - pct_start) / (1.0 - pct_start)
            factor         = 0.5 * (1 + np.cos(np.pi * phase_pct))
            optim.lr       = final_lr + (max_lr - final_lr) * factor
            optim.momentum = final_mom + (base_mom - final_mom) * factor

        w_hist.append(model.w.data[0, 0])
        b_hist.append(model.b.data[0, 0])

        y_pred = model(X)
        l      = loss(y_pred, Y)
        loss_hist.append(l)

        if pct >= pct_start:
            if l < best_loss - min_delta:
                best_loss  = l
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Adaptive early stopping at epoch {epoch + 1}: Cost did not improve for {patience} epochs (Best: {best_loss:.6f})")
                    break

        grad = loss.backward()
        model.backward(grad)
        optim.step()
        optim.zero_grad()

    w_hist    = np.array(w_hist)
    b_hist    = np.array(b_hist)
    loss_hist = np.array(loss_hist)
    print(f"Training complete. Final Cost: {loss_hist[-1]:.4f}")

    w_opt    = float(np.cov(X.squeeze(), Y.squeeze())[0, 1] / np.var(X))
    b_opt    = float(Y.mean() - w_opt * X.mean())
    min_cost = float(np.mean((w_opt * X + b_opt - Y) ** 2))

    fig = plt.figure(figsize=(18, 10))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title('BasicML - Linear Regression Dynamic Training')

    ax1   = fig.add_subplot(231)
    ax1.scatter(X, Y, color='blue', alpha=0.6, label='Training Data')
    line, = ax1.plot([], [], color='red', linewidth=2, label='Fitted Line')
    ax1.set_xlim(X.min() - 1, X.max() + 1)
    ax1.set_ylim(Y.min() - 2, Y.max() + 2)
    ax1.set_title("1. Linear Regression Fit")
    ax1.set_xlabel("X (Normalized)")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2        = fig.add_subplot(232)
    loss_line, = ax2.plot([], [], color='green', linewidth=2, label='MSE Loss')
    ax2.set_xlim(0, len(loss_hist))
    ax2.set_ylim(0, max(loss_hist) * 1.1)
    ax2.set_title("2. Learning Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Cost (MSE)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax5          = fig.add_subplot(233)
    loss_w_line, = ax5.plot([], [], color='purple', linewidth=2, label='Cost vs W')
    ax5.set_xlim(min(w_hist) - 0.5, max(w_hist) + 0.5)
    ax5.set_ylim(0, max(loss_hist) * 1.1)
    ax5.set_title("5. Cost vs Weight (w)")
    ax5.set_xlabel("Weight (w)")
    ax5.set_ylabel("Cost (MSE)")
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.6)

    ax3      = fig.add_subplot(234)
    w_margin = max(abs(w_hist.max() - w_hist.min()), 2.0) * 0.4
    b_margin = max(abs(b_hist.max() - b_hist.min()), 2.0) * 0.4

    w_min, w_max = min(w_hist.min(), w_opt) - w_margin, max(w_hist.max(), w_opt) + w_margin
    b_min, b_max = min(b_hist.min(), b_opt) - b_margin, max(b_hist.max(), b_opt) + b_margin

    w_vals = np.linspace(w_min, w_max, 50)
    b_vals = np.linspace(b_min, b_max, 50)
    W_grid, B_grid = np.meshgrid(w_vals, b_vals)
    Z_grid = np.zeros_like(W_grid)

    for i in range(len(w_vals)):
        for j in range(len(b_vals)):
            pred         = W_grid[j, i] * X + B_grid[j, i]
            Z_grid[j, i] = np.mean((pred - Y) ** 2)

    contour = ax3.contour(W_grid, B_grid, Z_grid, levels=np.linspace(min_cost, Z_grid.max(), 20), cmap='viridis', alpha=0.8)
    ax3.clabel(contour, inline=True, fontsize=8)
    ax3.plot([w_opt], [b_opt], marker='*', color='red', markersize=12, label=f'Global Min ({w_opt:.2f}, {b_opt:.2f})')

    path_line, = ax3.plot([], [], color='black', marker='o', markersize=3, linewidth=1, alpha=0.7, label='Momentum Path')

    ax3.set_xlim(w_min, w_max)
    ax3.set_ylim(b_min, b_max)
    ax3.set_title("3. 2D Gradient Path on Cost Surface")
    ax3.set_xlabel("Weight (w)")
    ax3.set_ylabel("Bias (b)")
    ax3.legend()

    ax4 = fig.add_subplot(235, projection='3d')
    ax4.plot_surface(W_grid, B_grid, Z_grid, cmap='viridis', alpha=0.6, edgecolor='none')
    path_line_3d, = ax4.plot([], [], [], color='black', marker='o', markersize=3, linewidth=2, label='Momentum Path')
    ax4.plot([w_opt], [b_opt], [min_cost], marker='*', color='red', markersize=12, label='Global Min')
    ax4.set_title("4. 3D Gradient Path")
    ax4.set_xlabel("Weight (w)")
    ax4.set_ylabel("Bias (b)")
    ax4.set_zlabel("Cost (MSE)")
    ax4.view_init(elev=30, azim=-60)

    def update(frame):
        y_pred_line = w_hist[frame] * X + b_hist[frame]
        line.set_data(X, y_pred_line)

        loss_line.set_data(range(frame + 1), loss_hist[:frame + 1])
        loss_w_line.set_data(w_hist[:frame + 1], loss_hist[:frame + 1])

        path_line.set_data(w_hist[:frame + 1], b_hist[:frame + 1])

        path_line_3d.set_data(w_hist[:frame + 1], b_hist[:frame + 1])
        path_line_3d.set_3d_properties(loss_hist[:frame + 1])

        ax1.set_title(f"1. Fit (Epoch {frame}): y = {w_hist[frame]:.2f}x + {b_hist[frame]:.2f}")
        ax2.set_title(f"2. Learning Curve: Cost = {loss_hist[frame]:.4f}")

        return line, loss_line, loss_w_line, path_line, path_line_3d

    print("Generating Animation...")
    anim = FuncAnimation(fig, update, frames=len(loss_hist), interval=5, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
