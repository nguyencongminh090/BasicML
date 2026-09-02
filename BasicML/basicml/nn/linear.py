from basicml.tensor import Tensor
from .module        import Module
from typing         import Optional
from numpy.typing   import ArrayLike
from .              import init
import numpy as np


class Linear(Module):
    def __init__(self, 
                in_features : int, 
                out_features: int, 
                init_type   : str  = 'xavier', 
                bias        : bool = True):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        self.in_features  = in_features
        self.out_features = out_features
        self.init_type    = init_type
        self.use_bias     = bias

        self.w: Tensor           = Tensor(np.zeros((in_features, out_features), dtype=np.float64), requires_grad=True)
        self.b: Optional[Tensor] = Tensor(np.zeros((1, out_features), dtype=np.float64), requires_grad=True) \
                                   if bias else None
        self.x: Optional[np.ndarray] = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_type == 'he':
            init.he_normal_(self.w)
        else:
            init.xavier_normal_(self.w)
        if self.b is not None:
            init.zeros_(self.b)

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        out    = self.x @ self.w.data
        if self.b is not None:
            out = out + self.b.data
        return out

    def parameters(self) -> list[Tensor]:
        return [self.w, self.b] if self.b is not None else [self.w]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")

        self.w.grad += self.x.T @ grad_output
        if self.b is not None:
            self.b.grad += np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.w.data.T
