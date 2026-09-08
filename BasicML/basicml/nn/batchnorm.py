from basicml.tensor import Tensor
from .module        import Module
from typing         import Optional
from numpy.typing   import ArrayLike
import numpy as np


class BatchNorm(Module):
    def __init__(self,
                num_features: int,
                eps         : float = 1e-5,
                momentum    : float = 0.1,
                affine      : bool  = True):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")

        self.num_features = num_features
        self.eps          = eps
        self.momentum     = momentum
        self.affine       = affine

        self.gamma: Optional[Tensor] = Tensor(np.ones((1, num_features), dtype=np.float64), requires_grad=True) \
                                       if affine else None
        self.beta : Optional[Tensor] = Tensor(np.zeros((1, num_features), dtype=np.float64), requires_grad=True) \
                                       if affine else None

        self.running_mean: np.ndarray = np.zeros((1, num_features), dtype=np.float64)
        self.running_var : np.ndarray = np.ones((1, num_features), dtype=np.float64)

        self.x_centered: Optional[np.ndarray] = None
        self.std_inv   : Optional[np.ndarray] = None
        self.x_hat     : Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"expected input of shape (batch, {self.num_features}), got {x.shape}")

        if self.training:
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var  = np.mean((x - batch_mean) ** 2, axis=0, keepdims=True)

            n                 = x.shape[0]
            unbiased_var      = batch_var * (n / (n - 1)) if n > 1 else batch_var
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var  = (1.0 - self.momentum) * self.running_var  + self.momentum * unbiased_var

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        self.x_centered = x - mean
        self.std_inv    = 1.0 / np.sqrt(var + self.eps)
        self.x_hat      = self.x_centered * self.std_inv

        out = self.x_hat
        if self.affine and self.gamma is not None and self.beta is not None:
            out = self.gamma.data * out + self.beta.data
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x_hat is None or self.x_centered is None or self.std_inv is None:
            raise RuntimeError("backward called before forward pass")

        if self.affine and self.gamma is not None and self.beta is not None:
            self.gamma.grad += np.sum(grad_output * self.x_hat, axis=0, keepdims=True)
            self.beta.grad  += np.sum(grad_output, axis=0, keepdims=True)
            grad_x_hat = grad_output * self.gamma.data
        else:
            grad_x_hat = grad_output

        if not self.training:
            return grad_x_hat * self.std_inv

        n         = grad_x_hat.shape[0]
        grad_var  = np.sum(grad_x_hat * self.x_centered, axis=0, keepdims=True) * -0.5 * self.std_inv ** 3
        grad_mean = np.sum(grad_x_hat * -self.std_inv, axis=0, keepdims=True) \
                    + grad_var * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)
        return grad_x_hat * self.std_inv \
               + grad_var * 2.0 * self.x_centered / n \
               + grad_mean / n

    def parameters(self) -> list[Tensor]:
        if self.affine and self.gamma is not None and self.beta is not None:
            return [self.gamma, self.beta]
        return []

    def __repr__(self):
        return f"BatchNorm({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})"
