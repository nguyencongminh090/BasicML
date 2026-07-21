from .optimizer     import Optimizer
from basicml.tensor import Tensor

class SGD(Optimizer):
    def __init__(self, parameters: list[Tensor], lr: float):
        super().__init__(parameters, lr)

    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
