from basicml.tensor import Tensor
from .module        import Module
from numpy.typing   import ArrayLike
import numpy as np


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: list[Module] = list(layers)

    def forward(self, X: ArrayLike) -> np.ndarray:
        out = np.asarray(X, dtype=np.float64)
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self, mode: bool = True):
        self.training = mode
        for layer in self.layers:
            layer.train(mode)
        return self

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def __repr__(self):
        inner = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {inner}\n)"
