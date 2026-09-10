from .module       import Module
from .conv         import _pair
from typing        import Optional
from numpy.typing  import ArrayLike
import numpy as np


class _WindowPool2D(Module):
    def __init__(self,
                kernel_size: int | tuple[int, int],
                stride     : Optional[int | tuple[int, int]] = None,
                padding    : int | tuple[int, int]           = 0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride      = _pair(stride) if stride is not None else self.kernel_size
        self.padding     = _pair(padding)
        self.x_shape    : Optional[tuple[int, ...]] = None
        self.padded_shape: Optional[tuple[int, ...]] = None

    def _output_size(self, height: int, width: int) -> tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        return (height - kh) // sh + 1, (width - kw) // sw + 1

    def _tap_slice(self, i: int, j: int, out_h: int, out_w: int) -> tuple[slice, slice]:
        sh, sw = self.stride
        return slice(i, i + sh * out_h, sh), slice(j, j + sw * out_w, sw)

    def __repr__(self):
        return f"{type(self).__name__}(kernel_size={self.kernel_size}, " \
               f"stride={self.stride}, padding={self.padding})"


class MaxPool2D(_WindowPool2D):
    def __init__(self,
                kernel_size: int | tuple[int, int],
                stride     : Optional[int | tuple[int, int]] = None,
                padding    : int | tuple[int, int]           = 0):
        super().__init__(kernel_size, stride, padding)
        self.argmax: Optional[np.ndarray] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        x            = np.asarray(X, dtype=np.float64)
        self.x_shape = x.shape
        ph, pw       = self.padding
        x_padded     = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-np.inf)
        self.padded_shape = x_padded.shape

        kh, kw       = self.kernel_size
        out_h, out_w = self._output_size(x_padded.shape[2], x_padded.shape[3])
        out          = np.full((x.shape[0], x.shape[1], out_h, out_w), -np.inf, dtype=np.float64)
        argmax       = np.zeros((x.shape[0], x.shape[1], out_h, out_w), dtype=np.intp)

        for i in range(kh):
            for j in range(kw):
                row, col = self._tap_slice(i, j, out_h, out_w)
                region   = x_padded[:, :, row, col]
                better   = region > out
                out      = np.where(better, region, out)
                argmax   = np.where(better, i * kw + j, argmax)

        self.argmax = argmax
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.argmax is None or self.x_shape is None or self.padded_shape is None:
            raise RuntimeError("backward called before forward pass")

        kh, kw       = self.kernel_size
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        dx_padded    = np.zeros(self.padded_shape, dtype=np.float64)

        for i in range(kh):
            for j in range(kw):
                row, col                   = self._tap_slice(i, j, out_h, out_w)
                selected                   = (self.argmax == i * kw + j)
                dx_padded[:, :, row, col] += grad_output * selected

        ph, pw = self.padding
        return dx_padded[:, :, ph:ph + self.x_shape[2], pw:pw + self.x_shape[3]]


class AvgPool2D(_WindowPool2D):
    def forward(self, X: ArrayLike) -> np.ndarray:
        x            = np.asarray(X, dtype=np.float64)
        self.x_shape = x.shape
        ph, pw       = self.padding
        x_padded     = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        self.padded_shape = x_padded.shape

        kh, kw       = self.kernel_size
        out_h, out_w = self._output_size(x_padded.shape[2], x_padded.shape[3])
        out          = np.zeros((x.shape[0], x.shape[1], out_h, out_w), dtype=np.float64)

        for i in range(kh):
            for j in range(kw):
                row, col = self._tap_slice(i, j, out_h, out_w)
                out     += x_padded[:, :, row, col]

        return out / (kh * kw)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x_shape is None or self.padded_shape is None:
            raise RuntimeError("backward called before forward pass")

        kh, kw       = self.kernel_size
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        dx_padded    = np.zeros(self.padded_shape, dtype=np.float64)
        share        = grad_output / (kh * kw)

        for i in range(kh):
            for j in range(kw):
                row, col                   = self._tap_slice(i, j, out_h, out_w)
                dx_padded[:, :, row, col] += share

        ph, pw = self.padding
        return dx_padded[:, :, ph:ph + self.x_shape[2], pw:pw + self.x_shape[3]]


class GlobalAvgPool2D(Module):
    def __init__(self):
        super().__init__()
        self.x_shape: Optional[tuple[int, ...]] = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        x            = np.asarray(X, dtype=np.float64)
        self.x_shape = x.shape
        return np.mean(x, axis=(2, 3), keepdims=True)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x_shape is None:
            raise RuntimeError("backward called before forward pass")
        n, c, h, w = self.x_shape
        return np.broadcast_to(grad_output / (h * w), self.x_shape).copy()

    def __repr__(self):
        return "GlobalAvgPool2D()"


class GlobalMaxPool2D(Module):
    def __init__(self):
        super().__init__()
        self.x_shape: Optional[tuple[int, ...]] = None
        self.argmax : Optional[np.ndarray]      = None

    def forward(self, X: ArrayLike) -> np.ndarray:
        x            = np.asarray(X, dtype=np.float64)
        self.x_shape = x.shape
        n, c, h, w   = x.shape
        flat         = x.reshape(n, c, h * w)
        self.argmax  = np.argmax(flat, axis=2)
        picked       = np.take_along_axis(flat, self.argmax[:, :, None], axis=2)
        return picked.reshape(n, c, 1, 1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x_shape is None or self.argmax is None:
            raise RuntimeError("backward called before forward pass")
        n, c, h, w = self.x_shape
        dx_flat    = np.zeros((n, c, h * w), dtype=np.float64)
        np.put_along_axis(dx_flat, self.argmax[:, :, None], grad_output.reshape(n, c, 1), axis=2)
        return dx_flat.reshape(n, c, h, w)

    def __repr__(self):
        return "GlobalMaxPool2D()"
