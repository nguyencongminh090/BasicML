# AI generated (refactored/authored with Claude Code)
"""Regularization vs overfitting: L1 / L2 on make_moons.

Idea: an over-capacity MLP trained on a SMALL and NOISY training set will overfit --
it chases the noise of individual points and grows a jagged decision boundary. That is
"high variance": accuracy on the training set is high but the train/val gap
(generalization gap) is large. Regularization (L1/L2) penalizes large weights -> a
simpler model -> the gap shrinks, at the cost of a slightly worse training fit.

The reported loss is the data-fit term only (BCE). The penalty R(theta) is printed
separately.

Note on hidden-layer activations: use ReLU, NOT Sigmoid.
Sigmoid hidden units saturate -> the gradient is scaled down by >=4x per layer -> the
data gradient becomes weak. The L2 weight-decay term (lambda_ * param, applied every
step regardless of the data) then dominates and drives every weight to 0 -> the output
sits at 0.5, BCE sits at ln 2 ~ 0.6931, accuracy 0.5 (a dead equilibrium). ReLU keeps
the gradient at 1 for active units, so it does not suffer from this.

Run directly: python BasicML/examples/train_regularization.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from basicml.datasets        import make_moons
from basicml.nn.sequential   import Sequential
from basicml.nn.linear       import Linear
from basicml.nn.activation   import ReLU, Sigmoid, Tanh
from basicml.nn.loss         import BinaryCrossEntropy
from basicml.optim.momentum  import Momentum
from basicml.regularization  import Regularizer, L1, L2

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------
SEED         = 0
N_TRAIN      = 40                     # deliberately small -> easy to overfit
N_VAL        = 800                    # large -> stable generalization estimate
NOISE        = 0.30                   # high noise -> "fake detail" for the model to chase

HIDDEN       = 64                     # over-capacity for this 2D problem
LEARN_RATE   = 0.05
MOMENTUM     = 0.9
EPOCHS       = 9_000
EVAL_EVERY   = 50                     # how often the learning curve is recorded
LOG_EVERY    = 1_000

LAMBDA_L2    = 0.05
LAMBDA_L1    = 0.01

SHOW_PLOT    = True
# ----------------------------------------------------------------------


def build_model() -> Sequential:
    """2 hidden layers, ReLU in the hidden layers (see module docstring for why not Sigmoid)."""
    return Sequential(
        Linear(2,      HIDDEN, init_type='he'), ReLU(),
        Linear(HIDDEN, HIDDEN, init_type='he'), ReLU(),
        Linear(HIDDEN, HIDDEN, init_type='he'), ReLU(),       
        Linear(HIDDEN, 1,      init_type='xavier'), Sigmoid(),
    )


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred >= 0.5).astype(np.float64) == y_true))


def total_penalty(reg: Regularizer, model: Sequential) -> float:
    return float(sum(reg.penalty(p) for p in model.parameters()))


class History:
    """Records train/val loss over epochs to draw the learning curve."""
    def __init__(self) -> None:
        self.epoch: list[int]   = []
        self.train: list[float] = []
        self.val:   list[float] = []

    def record(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.epoch.append(epoch)
        self.train.append(train_loss)
        self.val.append(val_loss)


class Result:
    def __init__(self, name: str, model: Sequential, hist: History,
                 tr_loss: float, val_loss: float, tr_acc: float, val_acc: float,
                 penalty: float | None) -> None:
        self.name      = name
        self.model     = model
        self.hist      = hist
        self.tr_loss   = tr_loss
        self.val_loss  = val_loss
        self.tr_acc    = tr_acc
        self.val_acc   = val_acc
        self.penalty   = penalty

    @property
    def loss_gap(self) -> float:
        return self.val_loss - self.tr_loss

    @property
    def acc_gap(self) -> float:
        return self.tr_acc - self.val_acc


def train(
    name : str,
    Xtr  : np.ndarray, ytr: np.ndarray,
    Xval : np.ndarray, yval: np.ndarray,
    reg  : Regularizer | None,
) -> Result:
    np.random.seed(SEED)                       # same weight init for every run
    model     = build_model()
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE,
                         regularizer=reg, momentum=MOMENTUM)
    hist      = History()

    print(f"\n=== {name} ===")
    for epoch in range(1, EPOCHS + 1):
        y_pred = model(Xtr)
        cost   = criterion(y_pred, ytr)

        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            hist.record(epoch, cost, criterion(model(Xval), yval))
        if epoch % LOG_EVERY == 0:
            print(f"[epoch] {epoch:5d} | [train BCE] {cost:.4f}")

    tr_pred, val_pred = model(Xtr), model(Xval)
    res = Result(
        name, model, hist,
        tr_loss  = criterion(tr_pred,  ytr),
        val_loss = criterion(val_pred, yval),
        tr_acc   = accuracy(tr_pred,   ytr),
        val_acc  = accuracy(val_pred,  yval),
        penalty  = total_penalty(reg, model) if reg is not None else None,
    )
    print(f"train : loss {res.tr_loss:.4f} | acc {res.tr_acc:.3f}")
    print(f"val   : loss {res.val_loss:.4f} | acc {res.val_acc:.3f}")
    print(f"gap   : loss {res.loss_gap:+.4f} | acc {res.acc_gap:+.3f}"
          "  (large gap = high variance = overfitting)")
    if res.penalty is not None:
        print(f"R(theta) = {res.penalty:.4f}")
    return res


def print_summary(results: list[Result]) -> None:
    print("\n" + "=" * 64)
    print(f"{'run':<22} {'train acc':>10} {'val acc':>9} {'loss gap':>10} {'R(theta)':>10}")
    print("-" * 64)
    for r in results:
        pen = f"{r.penalty:.4f}" if r.penalty is not None else "-"
        print(f"{r.name:<22} {r.tr_acc:>10.3f} {r.val_acc:>9.3f} {r.loss_gap:>+10.4f} {pen:>10}")
    print("=" * 64)
    print("large loss gap -> overfitting (high variance): fits train noise, generalizes poorly")
    print("regularization shrinks the gap, at the cost of slightly lower train acc (a bit more bias)")


def _boundary_grid(X: np.ndarray, model: Sequential):
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200),
    )
    zz = model(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, zz


def plot_boundary(ax, model: Sequential, X: np.ndarray, y: np.ndarray, title: str,
                  mark_wrong: bool = False) -> None:
    xx, yy, zz = _boundary_grid(X, model)
    ax.contourf(xx, yy, zz, levels=[0.0, 0.5, 1.0], cmap="bwr", alpha=0.25)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k", s=25)
    if mark_wrong:
        pred  = (model(X) >= 0.5).astype(np.float64)
        wrong = (pred != y).ravel()
        ax.scatter(X[wrong, 0], X[wrong, 1], marker="x", c="k", s=60, linewidths=1.5,
                   label=f"misclassified ({int(wrong.sum())})")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def plot_curves(ax, hist: History, title: str) -> None:
    ax.plot(hist.epoch, hist.train, label="train", color="tab:blue")
    ax.plot(hist.epoch, hist.val,   label="val",   color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE")
    ax.legend(fontsize=8)


def plot(results: list[Result],
         Xtr: np.ndarray, ytr: np.ndarray,
         Xval: np.ndarray, yval: np.ndarray) -> None:
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))
    for col, r in enumerate(results):
        plot_boundary(axes[0, col], r.model, Xtr, ytr,
                      f"{r.name}\ntrain  (acc {r.tr_acc:.3f})")
        plot_boundary(axes[1, col], r.model, Xval, yval,
                      f"val  (acc {r.val_acc:.3f})", mark_wrong=True)
        plot_curves(axes[2, col], r.hist,
                    f"learning curve  (val gap {r.loss_gap:+.3f})")
    fig.suptitle("Row 1: boundary over the training set (noisy).  "
                 "Row 2: same model over the held-out val set, x = misclassified.  "
                 "Row 3: train (blue) vs val (red) BCE -- val curving up = overfitting")
    plt.tight_layout()
    plt.show()


def main() -> None:
    Xtr,  ytr  = make_moons(N_TRAIN, noise=NOISE, random_state=SEED)
    Xval, yval = make_moons(N_VAL,   noise=NOISE, random_state=SEED + 1)
    ytr, yval  = ytr.astype(np.float64), yval.astype(np.float64)

    runs: list[tuple[str, Regularizer | None]] = [
        ("no regularization",        None),
        (f"L2 (lambda_={LAMBDA_L2})", L2(LAMBDA_L2)),
        (f"L1 (lambda_={LAMBDA_L1})", L1(LAMBDA_L1)),
    ]
    results = [train(name, Xtr, ytr, Xval, yval, reg) for name, reg in runs]
    print_summary(results)

    if SHOW_PLOT:
        plot(results, Xtr, ytr, Xval, yval)


if __name__ == "__main__":
    main()
