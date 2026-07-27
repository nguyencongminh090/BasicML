# AI generated

import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(suppress=True, precision=4)

from basicml.nn.models import LogisticRegressionModel
from basicml.nn.loss import BinaryCrossEntropy
from basicml.optim.momentum import Momentum

def generate_data(n: int, a: float, b: float):
    threshold = (a + b) / 4
    out = []
    for _ in range(n):
        x = random.uniform(a, b)
        out.append((x, 1 if x > threshold else 0))
    return out

def main():
    data = generate_data(100, 0, 10)
    X = np.array([[d[0] for d in data]]).reshape((-1, 1))
    Y = np.array([d[1] for d in data]).reshape((-1, 1))

    x_mean = X.mean()
    x_std  = X.std()
    X = (X - x_mean) / x_std

    epochs = 1500
    model  = LogisticRegressionModel(features=1)
    
    model.linear.w.data = np.array([[-1.0]])
    model.linear.b.data = np.array([[-2.0]])

    loss = BinaryCrossEntropy()

    base_lr, max_lr, final_lr    = 0.05, 0.4, 0.01
    max_mom, base_mom, final_mom = 0.95, 0.98, 0.95
    pct_start                    = 0.3

    optim      = Momentum(model.parameters(), lr=base_lr, momentum=max_mom)

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

        w_hist.append(model.parameters()[0].data[0, 0])
        b_hist.append(model.parameters()[1].data[0, 0])

        y_pred = model(X)
        l      = loss(y_pred, Y)
        loss_hist.append(l)

        grad = loss.backward()
        model.backward(grad)
        optim.step()
        optim.zero_grad()

    w_hist    = np.array(w_hist)
    b_hist    = np.array(b_hist)
    loss_hist = np.array(loss_hist)
    print(f"Training complete. Final Cost: {loss_hist[-1]:.4f}")

    w_opt = w_hist[-1]
    b_opt = b_hist[-1]
    min_cost = loss_hist[-1]

    fig = plt.figure(figsize=(10, 8))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title('BasicML - Logistic Regression 3D Cost Surface')

    ax4 = fig.add_subplot(111, projection='3d')

    w_margin = max(abs(w_hist.max() - w_hist.min()), 5.0) * 1.5
    b_margin = max(abs(b_hist.max() - b_hist.min()), 5.0) * 1.5

    w_min, w_max = min(w_hist.min(), w_opt) - w_margin, max(w_hist.max(), w_opt) + w_margin
    b_min, b_max = min(b_hist.min(), b_opt) - b_margin, max(b_hist.max(), b_opt) + b_margin

    w_vals = np.linspace(w_min, w_max, 50)
    b_vals = np.linspace(b_min, b_max, 50)
    W_grid, B_grid = np.meshgrid(w_vals, b_vals)
    Z_grid = np.zeros_like(W_grid)

    for i in range(len(w_vals)):
        for j in range(len(b_vals)):
            z = W_grid[j, i] * X + B_grid[j, i]
            pred = 1 / (1 + np.exp(-z))
            pred = np.clip(pred, 1e-15, 1 - 1e-15)
            Z_grid[j, i] = -np.mean(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))

    # Optional: clip Z so extreme costs don't stretch the Z axis too much
    Z_grid = np.clip(Z_grid, 0, min_cost + 15)

    ax4.plot_surface(W_grid, B_grid, Z_grid, cmap='viridis', alpha=0.6, edgecolor='none')
    path_line_3d, = ax4.plot([], [], [], color='black', marker='o', markersize=3, linewidth=2, label='Momentum Path')
    ax4.plot([w_opt], [b_opt], [min_cost], marker='*', color='red', markersize=12, label='End')
    
    # Make the plot less cubic (Wider X and Y, flatter Z)
    ax4.set_box_aspect((2, 2, 1))

    ax4.set_title("3D Gradient Path on Cost Surface (Logistic Regression)")
    ax4.set_xlabel("Weight (w)")
    ax4.set_ylabel("Bias (b)")
    ax4.set_zlabel("Cost (BCE)")
    ax4.view_init(elev=30, azim=-60)
    ax4.legend()

    def update(frame):
        path_line_3d.set_data(w_hist[:frame + 1], b_hist[:frame + 1])
        path_line_3d.set_3d_properties(loss_hist[:frame + 1])
        return path_line_3d,

    print("Generating Animation...")
    anim = FuncAnimation(fig, update, frames=len(loss_hist), interval=10, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
