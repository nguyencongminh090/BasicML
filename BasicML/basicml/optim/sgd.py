from .optimizer             import Optimizer
from typing                 import Optional
from basicml.regularization import Regularizer
from basicml.tensor         import Tensor

class SGD(Optimizer):
    def __init__(self, 
                 parameters : list[Tensor], 
                 lr         : float, 
                 regularizer: Optional[Regularizer] = None):
        super().__init__(parameters, lr, regularizer)

    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if self.regularizer is not None:
                    grad = grad + self.regularizer.grad(param)
                param.data -= self.lr * grad

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
