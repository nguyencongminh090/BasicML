from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass


class MSELoss(Loss):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return 1/(2*self.y_pred.shape[0]) * np.sum((self.y_pred - self.y_true)**2)

    def backward(self) -> np.ndarray:
        return self.y_pred - self.y_true