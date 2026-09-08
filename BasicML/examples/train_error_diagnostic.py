# AI generated (authored with Claude Code)
"""Model diagnostic: the train / train-dev / dev / test error ladder.

Trains the optimizer-comparison MLP on a "clean" slice of make_moons, then reads
four error rungs to localise what is limiting it (Ng's bias/variance recipe):

    human  ->  train  ->  train-dev  ->  dev  ->  test

  - train vs train-dev : both come from the training distribution, so a gap is
                         variance -- the model memorised the exact training rows.
  - train-dev vs dev   : dev/test are drawn from a noisier "target" distribution
                         here, so a gap is data mismatch, not variance.
  - dev vs test        : a gap means model / hyper-parameter choices overfit dev.

Run directly: python BasicML/examples/train_error_diagnostic.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from basicml.datasets      import make_moons, iter_minibatches
from basicml.diagnostics   import error_ladder
from basicml.nn.sequential import Sequential
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Sigmoid
from basicml.nn.loss       import BinaryCrossEntropy
from basicml.optim.adam    import Adam

SEED         = 0
HIDDEN       = 32
BATCH_SIZE   = 32
EPOCHS       = 120
TRAIN_NOISE  = 0.15   # training distribution
TARGET_NOISE = 0.35   # dev/test distribution -- deliberately shifted


def build_model() -> Sequential:
    return Sequential(
        Linear(2,      HIDDEN, init_type='he'),     ReLU(),
        Linear(HIDDEN, HIDDEN, init_type='he'),     ReLU(),
        Linear(HIDDEN, 1,      init_type='xavier'), Sigmoid(),
    )


def make_split(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples, noise=noise, random_state=seed)
    return X, y.astype(np.float64)


def train(model: Sequential, X: np.ndarray, y: np.ndarray) -> None:
    criterion = BinaryCrossEntropy()
    optimizer = Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        for Xb, yb in iter_minibatches(X, y, BATCH_SIZE, random_state=SEED + epoch):
            criterion(model(Xb), yb)
            model.backward(criterion.backward())
            optimizer.step()
            optimizer.zero_grad()


def main() -> None:
    np.random.seed(SEED)                       # deterministic weight init
    train_X,     train_y     = make_split(600, TRAIN_NOISE,  SEED)
    train_dev_X, train_dev_y = make_split(300, TRAIN_NOISE,  SEED + 1)
    dev_X,       dev_y       = make_split(300, TARGET_NOISE, SEED + 2)
    test_X,      test_y      = make_split(300, TARGET_NOISE, SEED + 3)

    model = build_model()
    train(model, train_X, train_y)

    ladder = error_ladder(
        model,
        train       = (train_X,     train_y),
        train_dev   = (train_dev_X, train_dev_y),
        dev         = (dev_X,       dev_y),
        test        = (test_X,      test_y),
        human_error = 0.02,
    )
    print(ladder.summary())


if __name__ == "__main__":
    main()
