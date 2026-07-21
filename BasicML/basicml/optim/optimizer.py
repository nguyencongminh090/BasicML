from abc import ABC, abstractmethod
from basicml.tensor import Tensor

class Optimizer(ABC):
    def __init__(self, parameters: list[Tensor], lr: float):
        self.parameters = parameters
        self.lr         = lr

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass
