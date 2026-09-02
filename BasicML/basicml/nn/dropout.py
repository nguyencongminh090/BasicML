from basicml.tensor import Tensor
from .module        import Module
from typing         import Optional
from numpy.typing   import ArrayLike
from .              import init
import numpy as np


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p    = p
        self.mask = None

    def forward(self, X):
        x = np.asarray(X, dtype=np.float64)
        if not self.training or self.p == 0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (np.random.rand(*x.shape) < keep) / keep
        return x * self.mask

    def backward(self, grad_output):
        return grad_output if self.mask is None else grad_output * self.mask

    def __repr__(self):
        return f"Dropout(p={self.p})"
