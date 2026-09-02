# AI generated (refactored/authored with Claude Code)
"""Logistic regression tren du lieu 1D sinh ngau nhien (nguong x > threshold).

Chay truc tiep: python BasicML/examples/train_logistic.py
"""
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from basicml.nn.sequential  import Sequential
from basicml.nn.linear      import Linear
from basicml.nn.activation  import Sigmoid
from basicml.nn.loss        import BinaryCrossEntropy
from basicml.optim.momentum import Momentum

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED           = 0
N_SAMPLES      = 100
X_RANGE        = (0.0, 10.0)          # khoang lay mau dac trung x
LABEL_FRACTION = 0.25                 # nhan = 1 khi x > lo + frac * (hi - lo)

NORMALIZE      = True                 # chuan hoa x -> hoi tu nhanh hon nhieu

LEARN_RATE     = 0.07
MOMENTUM       = 0.95
MAX_EPOCHS     = 20_000
LOSS_TARGET    = 0.05                  # du lieu tach roi tuyet doi -> loss giam theo log(epoch)
LOG_EVERY      = 2_000

SHOW_PLOT      = True
# ----------------------------------------------------------------------


def make_threshold_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng       = random.Random(SEED)
    lo, hi    = X_RANGE
    threshold = lo + LABEL_FRACTION * (hi - lo)

    x = np.array([rng.uniform(lo, hi) for _ in range(N_SAMPLES)]).reshape(-1, 1)
    y = (x > threshold).astype(np.float64)
    if NORMALIZE:
        x = (x - x.mean()) / x.std()
    return x, y


def train(x: np.ndarray, y: np.ndarray) -> tuple[Sequential, float, int]:
    model     = Sequential(Linear(in_features=1, out_features=1), Sigmoid())
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    cost = float("inf")
    for epoch in range(1, MAX_EPOCHS + 1):
        y_pred = model(x)
        cost   = criterion(y_pred, y)

        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

        if epoch % LOG_EVERY == 0:
            print(f"[epoch] {epoch:7d} | [loss] {cost:.6f}")
        if cost < LOSS_TARGET:
            break

    return model, cost, epoch


def report(model: Sequential, cost: float, epochs: int) -> None:
    weight, bias = (p.data.item() for p in model.parameters())
    print(f"converged after {epochs} epochs | loss = {cost:.6f}")
    print(f"w = {weight:.4f}, b = {bias:.4f}")
    print(f"f(x) = 1 / (1 + e^-({weight:.4f}x + ({bias:.4f})))")


def plot_fit(model: Sequential, x: np.ndarray, y: np.ndarray) -> None:
    x_line      = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_pred      = model(x)

    plt.scatter(x, y, c=y.ravel(), cmap="bwr")
    plt.plot(x_line, model(x_line), color="black")
    plt.vlines(x, y, y_pred, color="gray", linestyle="dashed", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.show()


def main() -> None:
    x, y                 = make_threshold_dataset()
    model, cost, epochs  = train(x, y)
    report(model, cost, epochs)
    if SHOW_PLOT:
        plot_fit(model, x, y)


if __name__ == "__main__":
    main()
