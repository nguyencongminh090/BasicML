from typing       import Optional
from numpy.typing import ArrayLike
import numpy as np



class Tensor:
    def __init__(self, data: ArrayLike, requires_grad: bool = False):
        self.data         : np.ndarray           = np.asarray(data)
        self.requires_grad: bool                 = requires_grad
        self.grad         : Optional[np.ndarray] = np.zeros_like(self.data) \
                                                   if self.requires_grad else None

    def zero_grad(self):
        if self.requires_grad and self.grad is not None:
            self.grad.fill(0.0)

    def _to_data(self, other: "Tensor | np.ndarray | float | int") -> np.ndarray:
        return other.data if isinstance(other, Tensor) else np.asarray(other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return Tensor(self.data.T)

    def __add__(self, other):
        return Tensor(self.data + self._to_data(other))

    def __sub__(self, other):
        return Tensor(self.data - self._to_data(other))

    def __mul__(self, other):
        return Tensor(self.data * self._to_data(other))

    def __truediv__(self, other):
        return Tensor(self.data / self._to_data(other))

    def __matmul__(self, other):
        return Tensor(self.data @ self._to_data(other))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad={self.requires_grad})"