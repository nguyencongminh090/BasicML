from .module      import Module
from typing        import Optional
from numpy.typing  import ArrayLike
import numpy as np


class Activation(Module):
    pass


class Sigmoid(Activation):
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-np.asarray(X)))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("backward called before forward pass")

        return grad_output * self.out * (1 - self.out)


class ReLU(Activation):
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.out = np.maximum(0, np.asarray(X))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("backward called before forward pass")
        return grad_output * (self.out > 0)


class Tanh(Activation):
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.out = np.tanh(np.asarray(X))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("backward called before forward pass")
        return grad_output * (1 - self.out ** 2)