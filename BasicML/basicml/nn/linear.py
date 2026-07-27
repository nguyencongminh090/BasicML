from basicml.tensor import Tensor
from .module        import Module
from typing         import Optional
from numpy.typing   import ArrayLike
import numpy as np


class Linear(Module):
    def __init__(self, features: int, init: str = 'xavier'):
        if init == 'he':
            scale = np.sqrt(2 / features)
        else:
            scale = np.sqrt(1 / features)
        self.w: Tensor               = Tensor(np.random.randn(features, 1) * scale, requires_grad=True)
        self.b: Tensor               = Tensor(np.zeros((1, 1)), requires_grad=True)
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X)
        return self.x @ self.w.data + self.b.data
    
    def parameters(self) -> list[Tensor]:
        return [self.w, self.b]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
            
        self.w.grad += self.x.T @ grad_output
        self.b.grad += np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.w.data.T
