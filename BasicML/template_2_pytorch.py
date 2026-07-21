from abc          import ABC, abstractmethod
from typing       import Dict
from numpy.typing import ArrayLike
import numpy  as np
import pandas as pd


class Module(ABC):
    @abstractmethod
    def forward(self, X: ArrayLike) -> np.ndarray:
        pass

    def __call__(self, X: ArrayLike) -> np.ndarray:
        return self.forward(X)

    @abstractmethod
    def parameters(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def gradients(self) -> Dict[str, np.ndarray]:
        pass


class Optimizer(ABC):
    def __init__(self, parameters: Dict[str, np.ndarray], lr: float):
        self.parameters = parameters
        self.lr         = lr

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass


class Loss(ABC):
    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass


class LinearRegression(Module):
    ...


def main():
    ...


if __name__ == '__main__':
    main()