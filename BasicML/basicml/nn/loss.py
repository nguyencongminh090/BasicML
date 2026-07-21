from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass
