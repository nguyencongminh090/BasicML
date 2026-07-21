from .optimizer     import Optimizer
from basicml.tensor import Tensor
import numpy as np

class Momentum(Optimizer):
    def __init__(self, parameters: list[Tensor], lr: float, momentum: float=0.9):
        super().__init__(parameters, lr)
        self.momentum   = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                self.velocities[i] = self.momentum * self.velocities[i] + param.grad
                param.data -= self.lr * self.velocities[i]

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
