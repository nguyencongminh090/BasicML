from .module       import Module
from typing        import Optional
from numpy.typing  import ArrayLike
import numpy as np


class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.x_shape: Optional[tuple[int, ...]] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        x            = np.asarray(X, dtype=np.float64)
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x_shape is None:
            raise RuntimeError("backward called before forward pass")
        return grad_output.reshape(self.x_shape)

    def __repr__(self):
        return "Flatten()"
