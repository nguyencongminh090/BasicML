# AI generated

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(suppress=True, precision=4)

from basicml.nn.linear import LinearRegression
from basicml.nn.loss import MSELoss
from basicml.optim.momentum import Momentum


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std  = np.std(X,  axis=0, keepdims=True)
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std


def main():
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data.csv')
    data      = pd.read_csv(data_path)

    X = data[['X']].values
    Y = data[['Y']].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    epochs = 1000
    model  = LinearRegression(features=1)
    loss   = MSELoss()

    base_lr, max_lr, final_lr    = 0.005, 0.08, 0.001
    max_mom, base_mom, final_mom = 0.95, 0.70, 0.875
    pct_start                    = 0.3

    optim      = Momentum(model.parameters(), lr=base_lr, momentum=max_mom)
    patience   = 15
    min_delta  = 1e-4
    best_loss  = float('inf')
    no_improve = 0

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

        y_pred = model(X_scaled)
        l      = loss(y_pred, Y)
        print(f'EPOCH: {epoch} | COST: {l:.6f}')

        if pct >= pct_start:
            if l < best_loss - min_delta:
                best_loss  = l
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'Adaptive early stopping at epoch {epoch + 1}: Cost did not improve for {patience} epochs (Best: {best_loss:.6f})')
                    break

        grad = loss.backward()
        model.backward(grad)
        optim.step()
        optim.zero_grad()

    w_norm = float(model.w.data[0, 0])
    b_norm = float(model.b.data[0, 0])
    print(f'w={w_norm:.4f}, b={b_norm:.4f}')

    X_new        = np.array([[120.0]])
    X_new_scaled = scaler.transform(X_new)
    y_new_pred   = model(X_new_scaled)

    print(f'Predict: x={X_new[0, 0]:.0f} y={y_new_pred.data[0, 0]:.4f}')


if __name__ == '__main__':
    main()
