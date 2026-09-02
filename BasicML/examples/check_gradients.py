# AI generated (refactored/authored with Claude Code)
"""Numerical gradient check cho Linear va mot vai MLP nho.

Chay truc tiep: python BasicML/examples/check_gradients.py
So sanh gradient giai tich (backward thu cong) voi sai phan trung tam.
Thoat voi ma 0 neu tat ca deu dat, 1 neu co truong hop sai.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from basicml.nn.linear     import Linear
from basicml.nn.sequential import Sequential
from basicml.nn.activation import ReLU, Tanh, Sigmoid
from basicml.nn.module     import Module

# --- CONFIG --------------------------------------------------------------
SEED         = 0
BATCH_SIZE   = 8
IN_FEATURES  = 4
EPSILON      = 1e-6      # buoc sai phan trung tam
TOLERANCE    = 1e-5      # nguong sai so tuong doi toi da chap nhan
# ----------------------------------------------------------------------


def numeric_grad(loss_fn, param: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    grad = np.zeros_like(param)
    it   = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx           = it.multi_index
        original      = param[idx]

        param[idx]    = original + eps
        loss_plus     = loss_fn()
        param[idx]    = original - eps
        loss_minus    = loss_fn()
        param[idx]    = original

        grad[idx]     = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(1e-8, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def check(name: str, model: Module, x: np.ndarray) -> bool:
    target = np.random.randn(*model(x).shape)

    def loss() -> float:
        return float(np.mean((model(x) - target) ** 2))

    out          = model(x)
    upstream_grad = (2.0 / out.size) * (out - target)
    model.backward(upstream_grad)

    worst = 0.0
    for param in model.parameters():
        assert param.grad is not None
        analytic = param.grad
        numeric  = numeric_grad(loss, param.data)
        worst    = max(worst, relative_error(analytic, numeric))

    passed = worst < TOLERANCE
    print(f"[{'OK  ' if passed else 'FAIL'}] {name:<22} max rel err = {worst:.2e}")
    return passed


def build_cases() -> list[tuple[str, Module]]:
    return [
        ("Linear",         Linear(IN_FEATURES, 3)),
        ("Linear no-bias",  Linear(IN_FEATURES, 3, bias=False)),
        ("MLP relu/tanh",   Sequential(Linear(IN_FEATURES, 6, init_type="he"), ReLU(),
                                       Linear(6, 5), Tanh(), Linear(5, 2))),
        ("MLP sigmoid",     Sequential(Linear(IN_FEATURES, 5), Sigmoid(),
                                       Linear(5, 1), Sigmoid())),
    ]


def main() -> None:
    np.random.seed(SEED)
    x        = np.random.randn(BATCH_SIZE, IN_FEATURES)
    all_pass = all(check(name, model, x) for name, model in build_cases())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
