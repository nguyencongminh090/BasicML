from basicml.tensor import Tensor
import numpy as np


def _as_array(tensor: Tensor | np.ndarray) -> np.ndarray:
    return tensor.data if isinstance(tensor, Tensor) else tensor


def _fan_in_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        fan_in = fan_out = int(np.prod(shape))
    else:
        fan_in, fan_out = shape[0], shape[1]
    return fan_in, fan_out


def xavier_normal_(tensor: Tensor | np.ndarray) -> Tensor | np.ndarray:
    data            = _as_array(tensor)
    fan_in, fan_out = _fan_in_out(data.shape)
    std             = np.sqrt(2.0 / (fan_in + fan_out))
    data[...]       = np.random.randn(*data.shape) * std
    return tensor

def he_normal_(tensor: Tensor | np.ndarray) -> Tensor | np.ndarray:
    data          = _as_array(tensor)
    fan_in, _     = _fan_in_out(data.shape)
    std           = np.sqrt(2.0 / fan_in)
    data[...]     = np.random.randn(*data.shape) * std
    return tensor

def zeros_(param: Tensor | np.ndarray) -> Tensor | np.ndarray:
    _as_array(param).fill(0.0)
    return param
