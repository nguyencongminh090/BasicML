from basicml.tensor import Tensor
from .module        import Module
from typing         import Optional
from numpy.typing   import ArrayLike
import numpy as np


class LinearRegression(Module):
    def __init__(self, features: int):
        self.w: Tensor               = Tensor(np.zeros((features, 1)), requires_grad=True)
        self.b: Tensor               = Tensor(np.zeros((1, 1)), requires_grad=True)
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X)
        return self.x @ self.w.data + self.b.data
    
    def parameters(self) -> list[Tensor]:
        return [self.w, self.b]

    def backward(self, grad_output: np.ndarray):
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
            
        m = self.x.shape[0]
        self.w.grad = (1 / m) * self.x.T @ grad_output
        self.b.grad = (1 / m) * np.sum(grad_output, axis=0, keepdims=True)
