from .optimizer             import Optimizer
from typing                 import Optional
from basicml.regularization import Regularizer
from basicml.tensor         import Tensor
import numpy as np

class RMSProp(Optimizer):
    def __init__(self,
                 parameters : list[Tensor],
                 lr         : float,
                 regularizer: Optional[Regularizer] = None,
                 rho        : float                 = 0.9,
                 eps        : float                 = 1e-8):
        super().__init__(parameters, lr, regularizer)
        self.rho        = rho
        self.eps        = eps
        self.mean_sq    = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if self.regularizer is not None:
                    grad = grad + self.regularizer.grad(param)
                self.mean_sq[i] = self.rho * self.mean_sq[i] + (1.0 - self.rho) * grad ** 2
                param.data -= self.lr * grad / (np.sqrt(self.mean_sq[i]) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
