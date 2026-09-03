from .optimizer             import Optimizer
from typing                 import Optional
from basicml.regularization import Regularizer
from basicml.tensor         import Tensor
import numpy as np

class Adagrad(Optimizer):
    def __init__(self,
                 parameters : list[Tensor],
                 lr         : float,
                 regularizer: Optional[Regularizer] = None,
                 eps        : float                 = 1e-8):
        super().__init__(parameters, lr, regularizer)
        self.eps            = eps
        self.accumulated_sq = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if self.regularizer is not None:
                    grad = grad + self.regularizer.grad(param)
                self.accumulated_sq[i] += grad ** 2
                param.data -= self.lr * grad / (np.sqrt(self.accumulated_sq[i]) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
