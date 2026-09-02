from .optimizer             import Optimizer
from typing                 import Optional
from basicml.regularization import Regularizer
from basicml.tensor         import Tensor
import numpy as np

class Momentum(Optimizer):
    def __init__(self, 
                 parameters : list[Tensor], 
                 lr         : float, 
                 regularizer: Optional[Regularizer] = None,
                 momentum   : float                 = 0.9):
        super().__init__(parameters, lr, regularizer)
        self.momentum    = momentum
        self.velocities  = [np.zeros_like(p.data) for p in self.parameters]
        self.regularizer = regularizer

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if self.regularizer is not None:
                    grad = grad + self.regularizer.grad(param)
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                param.data -= self.lr * self.velocities[i]

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
