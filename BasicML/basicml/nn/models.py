from basicml.tensor import Tensor
from .module        import Module
from .linear        import Linear
from .activation    import Sigmoid
from numpy.typing   import ArrayLike
import numpy as np


class LogisticRegressionModel(Module):
    def __init__(self, features: int):
        self.linear  = Linear(features)
        self.sigmoid = Sigmoid()

    def forward(self, X: ArrayLike) -> np.ndarray:
        logits = self.linear(X)
        return self.sigmoid(logits)
    
    def parameters(self) -> list[Tensor]:
        return self.linear.parameters()

    def backward(self, grad_output: np.ndarray):
        grad_sigmoid = self.sigmoid.backward(grad_output)
        return self.linear.backward(grad_sigmoid)
