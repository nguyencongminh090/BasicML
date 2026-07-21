from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

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
