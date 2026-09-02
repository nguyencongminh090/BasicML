# AI generated (refactored/authored with Claude Code)
"""Numerical gradient check for Linear and a few small MLPs.

Run directly: python BasicML/examples/check_gradients.py

Compares the analytic gradients (hand-written ``backward``) against a central
finite-difference estimate. Exits with code 0 if every case passes, 1 if any
case fails.

The central-difference estimate of ``dL/dp`` is
``(L(p + eps) - L(p - eps)) / (2 * eps)``, accurate to ``O(eps^2)``.
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
EPSILON      = 1e-6      # central-difference step
TOLERANCE    = 1e-5      # largest relative error still accepted
# ----------------------------------------------------------------------


def numeric_grad(loss_fn, param: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """Estimate ``dL/dparam`` by central finite differences.

    Perturbs each entry of ``param`` in place by ``±eps``, evaluates
    ``loss_fn`` at both points, and restores the original value.

    Args:
        loss_fn: Zero-argument callable returning the scalar loss for the
            current parameter values.
        param: Parameter array to differentiate; mutated and restored during
            the sweep.
        eps: Perturbation size.

    Returns:
        Array shaped like ``param`` holding the estimated gradient.
    """
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
    """Return the largest elementwise relative error between two arrays.

    Uses ``|a - b| / max(1e-8, |a| + |b|)`` so entries near zero do not blow
    the ratio up.

    Args:
        a: First array (e.g. analytic gradient).
        b: Second array (e.g. numeric gradient), same shape as ``a``.

    Returns:
        The maximum relative error over all entries.
    """
    denom = np.maximum(1e-8, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def check(name: str, model: Module, x: np.ndarray) -> bool:
    """Run one gradient check on ``model`` against a random MSE target.

    Computes the analytic gradients via ``model.backward`` and compares each
    parameter's gradient to its finite-difference estimate.

    Args:
        name: Label printed with the result.
        model: Module to check; its parameters must expose ``grad``.
        x: Input batch, shape ``(BATCH_SIZE, IN_FEATURES)``.

    Returns:
        ``True`` if the worst relative error is below ``TOLERANCE``.
    """
    target = np.random.randn(*model(x).shape)

    def loss() -> float:
        return float(np.mean((model(x) - target) ** 2))

    out           = model(x)
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
    """Build the list of ``(name, model)`` pairs to gradient-check.

    Returns:
        A list covering a bare ``Linear``, a bias-free ``Linear``, and two
        small MLPs mixing ReLU/Tanh/Sigmoid activations.
    """
    return [
        ("Linear",         Linear(IN_FEATURES, 3)),
        ("Linear no-bias",  Linear(IN_FEATURES, 3, bias=False)),
        ("MLP relu/tanh",   Sequential(Linear(IN_FEATURES, 6, init_type="he"), ReLU(),
                                       Linear(6, 5), Tanh(), Linear(5, 2))),
        ("MLP sigmoid",     Sequential(Linear(IN_FEATURES, 5), Sigmoid(),
                                       Linear(5, 1), Sigmoid())),
    ]


def main() -> None:
    """Run every gradient-check case and exit non-zero if any fails."""
    np.random.seed(SEED)
    x        = np.random.randn(BATCH_SIZE, IN_FEATURES)
    all_pass = all(check(name, model, x) for name, model in build_cases())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
