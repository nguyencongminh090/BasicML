from abc         import ABC, abstractmethod
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
        return 1/(self.y_pred.shape[0]) * (self.y_pred - self.y_true)


class AbsoluteLoss(Loss):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return 1/(self.y_pred.shape[0]) * np.sum(np.abs(self.y_pred - self.y_true))

    def backward(self) -> np.ndarray:
        return 1/(self.y_pred.shape[0]) * np.sign(self.y_pred - self.y_true)


class BinaryCrossEntropy(Loss):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        return -1/(self.y_pred.shape[0]) * \
            np.sum((self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred)))

    def backward(self) -> np.ndarray:
        return 1/(self.y_pred.shape[0]) * \
               ((self.y_pred - self.y_true) / (self.y_pred * (1- self.y_pred)))