# AI generated (refactored/authored with Claude Code)
"""Logistic regression on a randomly generated 1D threshold dataset.

Run directly: python BasicML/examples/train_logistic.py

Labels are ``1`` when ``x`` exceeds a fixed threshold. A ``Linear + Sigmoid``
model is trained with binary cross-entropy and the Adam optimizer, then the
fitted sigmoid is plotted against the data.
"""
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from basicml.nn.sequential import Sequential
from basicml.nn.linear     import Linear
from basicml.nn.activation import Sigmoid
from basicml.nn.loss       import BinaryCrossEntropy
from basicml.optim.adam    import Adam

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED           = 0
N_SAMPLES      = 100
X_RANGE        = (0.0, 10.0)          # range the feature x is sampled from
LABEL_FRACTION = 0.25                 # label = 1 when x > lo + frac * (hi - lo)

NORMALIZE      = True                 # standardizing x converges far faster

LEARN_RATE     = 0.1
MAX_EPOCHS     = 10**9
LOSS_TARGET    = 1e-8                 # perfectly separable data -> loss decays like log(epoch)
LOG_EVERY      = 250

SHOW_PLOT      = True
# ----------------------------------------------------------------------


def make_threshold_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate the 1D threshold classification dataset.

    Samples ``N_SAMPLES`` points uniformly from ``X_RANGE`` and labels each
    ``1`` when it lies above ``lo + LABEL_FRACTION * (hi - lo)``. When
    ``NORMALIZE`` is set, ``x`` is standardized to zero mean and unit variance.

    Returns:
        Tuple ``(x, y)`` of float64 arrays, each shaped ``(N_SAMPLES, 1)``.
    """
    rng       = random.Random(SEED)
    lo, hi    = X_RANGE
    threshold = lo + LABEL_FRACTION * (hi - lo)

    x = np.array([rng.uniform(lo, hi) for _ in range(N_SAMPLES)]).reshape(-1, 1)
    y = (x > threshold).astype(np.float64)
    if NORMALIZE:
        x = (x - x.mean()) / x.std()
    return x, y


def train(x: np.ndarray, y: np.ndarray) -> tuple[Sequential, float, int]:
    """Fit a ``Linear + Sigmoid`` model with binary cross-entropy.

    Stops when the loss drops below ``LOSS_TARGET`` or ``MAX_EPOCHS`` is
    reached, whichever comes first.

    Args:
        x: Input features, shape ``(N_SAMPLES, 1)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.

    Returns:
        Tuple ``(model, final_loss, epochs_run)``.
    """
    model     = Sequential(Linear(in_features=1, out_features=1), Sigmoid())
    criterion = BinaryCrossEntropy()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)

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
    """Print the learned parameters and the fitted decision function.

    Args:
        model: The trained ``Linear + Sigmoid`` model.
        cost: Final training loss.
        epochs: Number of epochs actually run.
    """
    weight, bias = (p.data.item() for p in model.parameters())
    print(f"converged after {epochs} epochs | loss = {cost:.6f}")
    print(f"w = {weight:.4f}, b = {bias:.4f}")
    print(f"f(x) = 1 / (1 + e^-({weight:.4f}x + ({bias:.4f})))")


def plot_fit(model: Sequential, x: np.ndarray, y: np.ndarray) -> None:
    """Scatter the data and overlay the fitted sigmoid probability curve.

    Args:
        model: The trained model.
        x: Input features, shape ``(N_SAMPLES, 1)``.
        y: Binary targets, shape ``(N_SAMPLES, 1)``.
    """
    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_pred = model(x)

    plt.scatter(x, y, c=y.ravel(), cmap="bwr")
    plt.plot(x_line, model(x_line), color="black")
    plt.vlines(x, y, y_pred, color="gray", linestyle="dashed", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.show()


def main() -> None:
    """Generate data, train the model, print a report, and plot the fit."""
    x, y                = make_threshold_dataset()
    model, cost, epochs = train(x, y)
    report(model, cost, epochs)
    if SHOW_PLOT:
        plot_fit(model, x, y)


if __name__ == "__main__":
    main()
