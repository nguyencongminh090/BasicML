# AI generated (refactored/authored with Claude Code)
"""Optimizer comparison: mini-batch gradient descent on make_moons.

Trains the *same* MLP (same weight init, same data, same mini-batch schedule)
with each optimizer in basicml.optim and plots the training-loss curves on one
axis. The point is to see the qualitative differences:

  - SGD / Momentum / Nesterov -- plain first-order; Momentum and Nesterov damp
    oscillations and accelerate along consistent directions.
  - Adagrad -- per-parameter step sizes, but the accumulated squared-gradient
    denominator only grows, so the effective learning rate decays toward zero.
  - RMSProp / Adadelta -- exponential average of squared gradients instead, so
    the step size adapts without vanishing.
  - Adam / AdamW -- RMSProp + momentum with bias correction; AdamW adds
    decoupled weight decay.

Mini-batching: each epoch iterates over shuffled batches (see
basicml.datasets.iter_minibatches), one optimizer step per batch.

Run directly: python BasicML/examples/train_optimizer_comparison.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from basicml.datasets      import make_moons, iter_minibatches
from basicml.nn.sequential import Sequential
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Sigmoid
from basicml.nn.loss       import BinaryCrossEntropy
from basicml.optim.optimizer import Optimizer
from basicml.optim.sgd       import SGD
from basicml.optim.momentum  import Momentum
from basicml.optim.nesterov  import Nesterov
from basicml.optim.adagrad   import Adagrad
from basicml.optim.rmsprop   import RMSProp
from basicml.optim.adadelta  import Adadelta
from basicml.optim.adam      import Adam
from basicml.optim.adamw     import AdamW

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED       = 0
N_SAMPLES  = 600
NOISE      = 0.20
HIDDEN     = 32
BATCH_SIZE = 32
EPOCHS     = 120
SHOW_PLOT  = True
# ----------------------------------------------------------------------

# name -> (parameters -> fresh Optimizer). A factory (not an instance) so every
# run rebuilds the optimizer's per-parameter state from scratch. Learning rates
# are hand-tuned so each optimizer roughly converges on this problem.
OPTIMIZERS: dict[str, Callable[[list], Optimizer]] = {
    "SGD":      lambda p: SGD(p,      lr=0.5),
    "Momentum": lambda p: Momentum(p, lr=0.1,  momentum=0.9),
    "Nesterov": lambda p: Nesterov(p, lr=0.1,  momentum=0.9),
    "Adagrad":  lambda p: Adagrad(p,  lr=0.1),
    "RMSProp":  lambda p: RMSProp(p,  lr=0.01, rho=0.9),
    "Adadelta": lambda p: Adadelta(p, lr=1.0,  rho=0.95),
    "Adam":     lambda p: Adam(p,     lr=0.01),
    "AdamW":    lambda p: AdamW(p,    lr=0.01, weight_decay=0.01),
}


def build_model() -> Sequential:
    return Sequential(
        Linear(2,      HIDDEN, init_type='he'),     ReLU(),
        Linear(HIDDEN, HIDDEN, init_type='he'),     ReLU(),
        Linear(HIDDEN, 1,      init_type='xavier'), Sigmoid(),
    )


def train_one(name: str, make_optimizer: Callable[[list], Optimizer],
              X: np.ndarray, y: np.ndarray) -> list[float]:
    np.random.seed(SEED)                       # identical weight init every run
    model     = build_model()
    criterion = BinaryCrossEntropy()
    optimizer = make_optimizer(model.parameters())
    history: list[float] = []

    print(f"\n=== {name} ===")
    for epoch in range(1, EPOCHS + 1):
        for Xb, yb in iter_minibatches(X, y, BATCH_SIZE, random_state=SEED + epoch):
            y_pred = model(Xb)
            criterion(y_pred, yb)
            model.backward(criterion.backward())
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = criterion(model(X), y)
        history.append(epoch_loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"[epoch] {epoch:4d} | [full-set BCE] {epoch_loss:.4f}")
    return history


def print_summary(histories: dict[str, list[float]]) -> None:
    print("\n" + "=" * 44)
    print(f"{'optimizer':<12} {'final BCE':>12} {'epoch<0.30':>14}")
    print("-" * 44)
    for name, hist in histories.items():
        reached = next((i + 1 for i, v in enumerate(hist) if v < 0.30), None)
        print(f"{name:<12} {hist[-1]:>12.4f} {str(reached):>14}")
    print("=" * 44)


def plot(histories: dict[str, list[float]]) -> None:
    plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        plt.plot(range(1, EPOCHS + 1), hist, label=name, linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("training BCE (log scale)")
    plt.title("Optimizer comparison -- mini-batch GD on make_moons")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    X, y = make_moons(N_SAMPLES, noise=NOISE, random_state=SEED)
    y    = y.astype(np.float64)

    histories = {name: train_one(name, factory, X, y)
                 for name, factory in OPTIMIZERS.items()}

    print_summary(histories)
    if SHOW_PLOT:
        plot(histories)


if __name__ == "__main__":
    main()
