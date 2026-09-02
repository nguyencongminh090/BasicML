"""Numerical gradient check cho Linear va mot MLP nho.

Chay truc tiep: python BasicML/examples/check_gradients.py
So sanh gradient giai tich (backward thu cong) voi sai phan trung tam.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from basicml.nn.linear     import Linear
from basicml.nn.sequential import Sequential
from basicml.nn.activation import ReLU, Tanh


def numeric_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it   = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx        = it.multi_index
        old        = x[idx]
        x[idx]     = old + eps
        plus       = f()
        x[idx]     = old - eps
        minus      = f()
        x[idx]     = old
        grad[idx]  = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def rel_err(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def check(name, model, x):
    y_pred = model(x)
    target = np.random.randn(*y_pred.shape)

    def loss():
        return float(np.mean((model(x) - target) ** 2))

    out  = model(x)
    grad = (2.0 / out.size) * (out - target)
    model.backward(grad)

    worst = 0.0
    for p in model.parameters():
        ng    = numeric_grad(loss, p.data)
        worst = max(worst, rel_err(p.grad, ng))

    status = "OK  " if worst < 1e-5 else "FAIL"
    print(f"[{status}] {name:<22} max rel err = {worst:.2e}")
    return worst < 1e-5


def main():
    np.random.seed(0)
    x = np.random.randn(8, 4)

    ok = True
    ok &= check("Linear"        , Linear(4, 3),                                    x)
    ok &= check("Linear no-bias", Linear(4, 3, bias=False),                        x)
    ok &= check("MLP relu/tanh" , Sequential(Linear(4, 6, init_type="he"), ReLU(),
                                             Linear(6, 5), Tanh(), Linear(5, 2)),  x)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
