from abc            import ABC, abstractmethod
from basicml.tensor import Tensor
from numpy.typing   import ArrayLike
import numpy as np


class Module(ABC):
    def __init__(self):
        self.training: bool = True

    @abstractmethod
    def forward(self, X: ArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, X: ArrayLike) -> np.ndarray:
        return self.forward(X)

    def parameters(self) -> list[Tensor]:
        return []
