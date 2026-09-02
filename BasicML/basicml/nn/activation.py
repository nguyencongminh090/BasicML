from basicml.tensor import Tensor
from .module        import Module
from typing         import Optional
from numpy.typing   import ArrayLike
import numpy as np


class Activation(Module):
    pass


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.out: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.out = np.tanh(np.asarray(X))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("backward called before forward pass")
        return grad_output * (1 - self.out ** 2)


class Identity(Activation):
    def forward(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope          = negative_slope
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return np.where(self.x > 0, self.x, self.negative_slope * self.x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        local = np.where(self.x > 0, 1.0, self.negative_slope)
        return grad_output * local


class PReLU(Activation):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        if num_parameters <= 0:
            raise ValueError("num_parameters must be positive")

        self.num_parameters = num_parameters
        self.a: Tensor      = Tensor(np.full((num_parameters,), init, dtype=np.float64),
                                     requires_grad=True)
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return np.where(self.x > 0, self.x, self.a.data * self.x)

    def parameters(self) -> list[Tensor]:
        return [self.a]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")

        neg    = self.x <= 0
        grad_a = grad_output * np.where(neg, self.x, 0.0)
        if self.num_parameters == 1:
            self.a.grad += np.array([grad_a.sum()])
        else:
            axes = tuple(range(grad_a.ndim - 1))
            self.a.grad += grad_a.sum(axis=axes)

        local = np.where(self.x > 0, 1.0, self.a.data)
        return grad_output * local


class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha                   = alpha
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return np.where(self.x > 0, self.x, self.alpha * np.expm1(self.x))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        local = np.where(self.x > 0, 1.0, self.alpha * np.exp(self.x))
        return grad_output * local


class SELU(Activation):
    ALPHA: float = 1.6732632423543772
    SCALE: float = 1.0507009873554805

    def __init__(self):
        super().__init__()
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return self.SCALE * np.where(self.x > 0, self.x, self.ALPHA * np.expm1(self.x))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        local = self.SCALE * np.where(self.x > 0, 1.0, self.ALPHA * np.exp(self.x))
        return grad_output * local


class Softplus(Activation):
    def __init__(self):
        super().__init__()
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return np.logaddexp(0.0, self.x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        return grad_output * (1 / (1 + np.exp(-self.x)))


class GELU(Activation):
    _C: float = 0.7978845608028654
    _A: float = 0.044715

    def __init__(self):
        super().__init__()
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        inner  = self._C * (self.x + self._A * self.x ** 3)
        return 0.5 * self.x * (1 + np.tanh(inner))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        x     = self.x
        u     = self._C * (x + self._A * x ** 3)
        g     = np.tanh(u)
        du_dx = self._C * (1 + 3 * self._A * x ** 2)
        local = 0.5 * (1 + g) + 0.5 * x * (1 - g ** 2) * du_dx
        return grad_output * local


class Swish(Activation):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta                      = beta
        self.x  : Optional[np.ndarray] = None
        self.sig: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x   = np.asarray(X, dtype=np.float64)
        self.sig = 1 / (1 + np.exp(-self.beta * self.x))
        return self.x * self.sig

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None or self.sig is None:
            raise RuntimeError("backward called before forward pass")
        local = self.sig + self.beta * self.x * self.sig * (1 - self.sig)
        return grad_output * local


class Mish(Activation):
    def __init__(self):
        super().__init__()
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return self.x * np.tanh(np.logaddexp(0.0, self.x))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        sp    = np.logaddexp(0.0, self.x)
        t     = np.tanh(sp)
        sig   = 1 / (1 + np.exp(-self.x))
        local = t + self.x * (1 - t ** 2) * sig
        return grad_output * local


class Hardtanh(Activation):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val")
        self.min_val                 = min_val
        self.max_val                 = max_val
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return np.clip(self.x, self.min_val, self.max_val)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        mask = (self.x > self.min_val) & (self.x < self.max_val)
        return grad_output * mask


class Hardsigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.x: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        self.x = np.asarray(X, dtype=np.float64)
        return np.clip(self.x / 6.0 + 0.5, 0.0, 1.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("backward called before forward pass")
        mask = (self.x > -3.0) & (self.x < 3.0)
        return grad_output * mask / 6.0


class Softmax(Activation):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis                      = axis
        self.out: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        x        = np.asarray(X, dtype=np.float64)
        shifted  = x - np.max(x, axis=self.axis, keepdims=True)
        exp      = np.exp(shifted)
        self.out = exp / np.sum(exp, axis=self.axis, keepdims=True)
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("backward called before forward pass")
        dot = np.sum(grad_output * self.out, axis=self.axis, keepdims=True)
        return self.out * (grad_output - dot)
