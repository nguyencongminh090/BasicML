from basicml.tensor import Tensor
from .module        import Module
from .conv          import Conv2D
from .pool          import MaxPool2D
from .flatten       import Flatten
from .linear        import Linear
from .activation    import ReLU, Sigmoid
from numpy.typing   import ArrayLike
import numpy as np


class CNNModel(Module):
    def __init__(self,
                input_shape  : tuple[int, int, int],
                num_classes  : int = 1,
                conv_channels: int = 8,
                kernel_size  : int = 3,
                pool_size    : int = 2,
                init_type    : str = 'he'):
        super().__init__()
        channels, height, width = input_shape
        if height % pool_size != 0 or width % pool_size != 0:
            raise ValueError("input height and width must be divisible by pool_size")

        self.conv    = Conv2D(channels, conv_channels, kernel_size,
                              padding=kernel_size // 2, init_type=init_type)
        self.relu    = ReLU()
        self.pool    = MaxPool2D(pool_size)
        self.flatten = Flatten()
        self.linear  = Linear(conv_channels * (height // pool_size) * (width // pool_size),
                              num_classes, init_type='xavier')
        self.sigmoid = Sigmoid()

    def forward(self, X: ArrayLike) -> np.ndarray:
        out = self.conv(X)
        out = self.relu(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = self.sigmoid.backward(grad_output)
        grad = self.linear.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.conv.backward(grad)
        return grad

    def parameters(self) -> list[Tensor]:
        return self.conv.parameters() + self.linear.parameters()

    def __repr__(self):
        inner = ",\n  ".join(repr(layer) for layer in
                             (self.conv, self.relu, self.pool, self.flatten, self.linear, self.sigmoid))
        return f"CNNModel(\n  {inner}\n)"
