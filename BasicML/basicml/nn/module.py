from abc import ABC, abstractmethod
from typing import Dict
from numpy.typing import ArrayLike
import numpy as np

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
