# AI generated (refactored/authored with Claude Code)
"""Linear regression tren BasicML/data.csv.

Chay truc tiep: python BasicML/examples/train_linear.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from basicml.nn.linear     import Linear
from basicml.nn.loss       import MSELoss
from basicml.optim.momentum import Momentum

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ---------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(REPO_ROOT, "data.csv")
X_COLUMNS  = ["X"]
Y_COLUMNS  = ["Y"]

EPOCHS     = 200
LEARN_RATE = 0.01
MOMENTUM   = 0.7
# -------------------------------------------------------------------------


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    x     = frame[X_COLUMNS].to_numpy(dtype=np.float64)
    y     = frame[Y_COLUMNS].to_numpy(dtype=np.float64)
    return x, y


def train(x: np.ndarray, y: np.ndarray) -> Linear:
    model     = Linear(in_features=x.shape[1], out_features=y.shape[1])
    criterion = MSELoss()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    for epoch in range(EPOCHS):
        y_pred = model(x)
        cost   = criterion(y_pred, y)

        model.backward(criterion.backward())
        optimizer.step()
        optimizer.zero_grad()

        print(f"EPOCH: {epoch:4d} | COST: {cost:.6f}")

    return model


def main() -> None:
    x, y  = load_dataset(DATA_PATH)
    model = train(x, y)

    weight, bias = (p.data.reshape(-1)[0] for p in model.parameters())
    print(f"f(x) = {weight:.4f}x + {bias:.4f}")


if __name__ == "__main__":
    main()
