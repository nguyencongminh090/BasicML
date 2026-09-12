from .optimizer             import Optimizer
from typing                 import Optional
from basicml.regularization import Regularizer
from basicml.tensor         import Tensor
import numpy as np


def newton_schulz5(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(Optimizer):
    def __init__(self,
                 parameters : list[Tensor],
                 lr         : float,
                 regularizer: Optional[Regularizer] = None,
                 momentum   : float                 = 0.95,
                 nesterov   : bool                  = True,
                 ns_steps   : int                   = 5,
                 eps        : float                 = 1e-7):
        super().__init__(parameters, lr, regularizer)
        self.momentum   = momentum
        self.nesterov   = nesterov
        self.ns_steps   = ns_steps
        self.eps        = eps
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if self.regularizer is not None:
                    grad = grad + self.regularizer.grad(param)
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                update = self.momentum * self.velocities[i] + grad if self.nesterov \
                         else self.velocities[i]

                if update.ndim >= 2:
                    shape  = update.shape
                    matrix = update.reshape(shape[0], -1)
                    ortho  = newton_schulz5(matrix, self.ns_steps, self.eps)
                    scale  = np.sqrt(max(1.0, matrix.shape[0] / matrix.shape[1]))
                    param.data -= self.lr * scale * ortho.reshape(shape)
                else:
                    param.data -= self.lr * update

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
