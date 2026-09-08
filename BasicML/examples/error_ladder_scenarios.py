# AI generated (authored with Claude Code)
"""Error-ladder diagnostic across five contrasting scenarios.

`train_error_diagnostic.py` shows the ladder once; this script builds five
setups whose limiting factor is different each time, so you can watch
`ErrorLadder.diagnosis()` change:

  1. underfit        -- logistic regression on moons -> avoidable bias
  2. overfit         -- big MLP, 25 training rows    -> variance
  3. data mismatch   -- train/target noise differ    -> data mismatch
  4. well-specified  -- capacity and data both fine  -> flat ladder
  5. regression      -- MSE error_fn, linear target  -> flat ladder

Run directly: python BasicML/examples/error_ladder_scenarios.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable

import numpy as np

from basicml.datasets      import make_moons, make_parabola_regression, iter_minibatches
from basicml.diagnostics   import ErrorLadder, error_ladder
from basicml.nn.module     import Module
from basicml.nn.sequential import Sequential
from basicml.nn.linear     import Linear
from basicml.nn.activation import ReLU, Sigmoid
from basicml.nn.loss       import BinaryCrossEntropy, MSELoss, Loss
from basicml.optim.adam    import Adam
from basicml.optim.optimizer import Optimizer

SEED = 0

Split        = tuple[np.ndarray, np.ndarray]
OptimFactory = Callable[[list], Optimizer]


def moons(n_samples: int, noise: float, seed: int) -> Split:
    X, y = make_moons(n_samples, noise=noise, random_state=seed)
    return X, y.astype(np.float64)


def parabola(n_samples: int, seed: int) -> Split:
    return make_parabola_regression(n_samples, target_noise=0.5, random_state=seed)


def fit(model: Module, X: np.ndarray, y: np.ndarray, criterion: Loss,
        make_optimizer: OptimFactory, epochs: int, batch_size: int) -> Module:
    optimizer = make_optimizer(model.parameters())
    model.train()
    for epoch in range(1, epochs + 1):
        for Xb, yb in iter_minibatches(X, y, batch_size, random_state=SEED + epoch):
            criterion(model(Xb), yb)
            model.backward(criterion.backward())
            optimizer.step()
            optimizer.zero_grad()
    return model


def mse_error(model: Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    pred = np.asarray(model(X)).reshape(-1)
    true = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(np.mean((pred - true) ** 2))


def underfit() -> ErrorLadder:
    np.random.seed(SEED)
    model = fit(
        Sequential(Linear(2, 1, init_type="xavier"), Sigmoid()),
        *moons(600, 0.15, SEED),
        BinaryCrossEntropy(), lambda p: Adam(p, lr=0.05), epochs=150, batch_size=32,
    )
    return error_ladder(
        model,
        train       = moons(600, 0.15, SEED),
        train_dev   = moons(300, 0.15, SEED + 1),
        dev         = moons(300, 0.15, SEED + 2),
        test        = moons(300, 0.15, SEED + 3),
        human_error = 0.02,
    )


def overfit() -> ErrorLadder:
    np.random.seed(SEED)
    model = fit(
        Sequential(
            Linear(2,   128, init_type="he"), ReLU(),
            Linear(128, 128, init_type="he"), ReLU(),
            Linear(128, 1,   init_type="xavier"), Sigmoid(),
        ),
        *moons(25, 0.30, SEED),
        BinaryCrossEntropy(), lambda p: Adam(p, lr=0.01), epochs=800, batch_size=25,
    )
    return error_ladder(
        model,
        train       = moons(25,  0.30, SEED),
        train_dev   = moons(300, 0.30, SEED + 1),
        dev         = moons(300, 0.30, SEED + 2),
        test        = moons(300, 0.30, SEED + 3),
        human_error = 0.05,
    )


def data_mismatch() -> ErrorLadder:
    np.random.seed(SEED)
    model = fit(
        Sequential(
            Linear(2,  32, init_type="he"), ReLU(),
            Linear(32, 32, init_type="he"), ReLU(),
            Linear(32, 1,  init_type="xavier"), Sigmoid(),
        ),
        *moons(600, 0.15, SEED),
        BinaryCrossEntropy(), lambda p: Adam(p, lr=0.01), epochs=150, batch_size=32,
    )
    return error_ladder(
        model,
        train       = moons(600, 0.15, SEED),
        train_dev   = moons(300, 0.15, SEED + 1),
        dev         = moons(300, 0.40, SEED + 2),
        test        = moons(300, 0.40, SEED + 3),
        human_error = 0.02,
    )


def well_specified() -> ErrorLadder:
    np.random.seed(SEED)
    model = fit(
        Sequential(
            Linear(2,  32, init_type="he"), ReLU(),
            Linear(32, 32, init_type="he"), ReLU(),
            Linear(32, 1,  init_type="xavier"), Sigmoid(),
        ),
        *moons(600, 0.15, SEED),
        BinaryCrossEntropy(), lambda p: Adam(p, lr=0.01), epochs=150, batch_size=32,
    )
    return error_ladder(
        model,
        train       = moons(600, 0.15, SEED),
        train_dev   = moons(300, 0.15, SEED + 1),
        dev         = moons(300, 0.15, SEED + 2),
        test        = moons(300, 0.15, SEED + 3),
        human_error = 0.02,
    )


def regression() -> ErrorLadder:
    np.random.seed(SEED)
    model = fit(
        Sequential(Linear(2, 1, init_type="xavier")),
        *parabola(600, SEED),
        MSELoss(), lambda p: Adam(p, lr=0.05), epochs=200, batch_size=32,
    )
    return error_ladder(
        model,
        train       = parabola(600, SEED),
        train_dev   = parabola(300, SEED + 1),
        dev         = parabola(300, SEED + 2),
        test        = parabola(300, SEED + 3),
        human_error = 0.5 ** 2,   # Bayes MSE == target-noise variance
        error_fn    = mse_error,
    )


SCENARIOS: dict[str, Callable[[], ErrorLadder]] = {
    "1. underfit (logistic regression on moons)": underfit,
    "2. overfit (128-wide MLP, 25 training rows)": overfit,
    "3. data mismatch (train noise 0.15, dev/test noise 0.40)": data_mismatch,
    "4. well-specified (capacity and data both adequate)": well_specified,
    "5. regression (linear model, MSE error_fn)": regression,
}


# ~2 standard errors of an error rate on a 300-row split
# (sqrt(0.12 * 0.88 / 300) ~ 0.019); gaps under this are sampling noise.
TOLERANCE = 0.035


def main() -> None:
    for title, scenario in SCENARIOS.items():
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print(scenario().summary(tolerance=TOLERANCE))


if __name__ == "__main__":
    main()
