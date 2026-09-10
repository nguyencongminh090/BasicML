from basicml.tensor import Tensor
from .module        import Module
from typing         import Optional
from numpy.typing   import ArrayLike
from .              import init
import numpy as np


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class Conv2D(Module):
    def __init__(self,
                in_channels : int,
                out_channels: int,
                kernel_size : int | tuple[int, int],
                stride      : int | tuple[int, int] = 1,
                padding     : int | tuple[int, int] = 0,
                dilation    : int | tuple[int, int] = 1,
                init_type   : str  = 'he',
                bias        : bool = True,
                groups      : int  = 1):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if groups != 1:
            raise ValueError("Conv2D supports groups=1 only")

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = _pair(kernel_size)
        self.stride       = _pair(stride)
        self.padding      = _pair(padding)
        self.dilation     = _pair(dilation)
        self.init_type    = init_type
        self.use_bias     = bias

        kh, kw = self.kernel_size
        self.w: Tensor           = Tensor(np.zeros((out_channels, in_channels, kh, kw), dtype=np.float64), requires_grad=True)
        self.b: Optional[Tensor] = Tensor(np.zeros((out_channels,), dtype=np.float64), requires_grad=True) \
                                   if bias else None
        self.x_padded: Optional[np.ndarray] = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_type == 'xavier':
            init.xavier_normal_(self.w)
        else:
            init.he_normal_(self.w)
        if self.b is not None:
            init.zeros_(self.b)

    def _output_size(self, height: int, width: int) -> tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        out_h  = (height - dh * (kh - 1) - 1) // sh + 1
        out_w  = (width  - dw * (kw - 1) - 1) // sw + 1
        return out_h, out_w

    def _tap_slice(self, i: int, j: int, out_h: int, out_w: int) -> tuple[slice, slice]:
        sh, sw = self.stride
        dh, dw = self.dilation
        row    = slice(i * dh, i * dh + sh * out_h, sh)
        col    = slice(j * dw, j * dw + sw * out_w, sw)
        return row, col

    def forward(self, X: ArrayLike) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected input of shape (batch, {self.in_channels}, H, W), got {x.shape}")

        ph, pw        = self.padding
        self.x_padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))

        kh, kw        = self.kernel_size
        out_h, out_w  = self._output_size(self.x_padded.shape[2], self.x_padded.shape[3])
        out           = np.zeros((x.shape[0], self.out_channels, out_h, out_w), dtype=np.float64)

        for i in range(kh):
            for j in range(kw):
                row, col = self._tap_slice(i, j, out_h, out_w)
                patch    = self.x_padded[:, :, row, col]
                out     += np.einsum('nihw,oi->nohw', patch, self.w.data[:, :, i, j])

        if self.b is not None:
            out += self.b.data[None, :, None, None]
        return out

    def parameters(self) -> list[Tensor]:
        return [self.w, self.b] if self.b is not None else [self.w]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x_padded is None or self.w.grad is None:
            raise RuntimeError("backward called before forward pass")

        if self.b is not None and self.b.grad is not None:
            self.b.grad += np.sum(grad_output, axis=(0, 2, 3))

        kh, kw       = self.kernel_size
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        dx_padded    = np.zeros_like(self.x_padded)
        w_grad       = self.w.grad

        for i in range(kh):
            for j in range(kw):
                row, col                  = self._tap_slice(i, j, out_h, out_w)
                patch                     = self.x_padded[:, :, row, col]
                w_grad[:, :, i, j]       += np.einsum('nohw,nihw->oi', grad_output, patch)
                dx_padded[:, :, row, col] += np.einsum('nohw,oi->nihw', grad_output, self.w.data[:, :, i, j])

        ph, pw = self.padding
        h_end  = dx_padded.shape[2] - ph
        w_end  = dx_padded.shape[3] - pw
        return dx_padded[:, :, ph:h_end, pw:w_end]

    def __repr__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " \
               f"dilation={self.dilation}, bias={self.use_bias}, init_type={self.init_type!r})"
