from abc                    import ABC, abstractmethod
from typing                 import Optional
from basicml.regularization import Regularizer
from basicml.tensor         import Tensor

class Optimizer(ABC):
    def __init__(self, 
                 parameters  : list[Tensor], 
                 lr          : float,
                 regularizer: Optional[Regularizer] = None):
        self.parameters  = parameters
        self.lr          = lr
        self.regularizer = regularizer

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass
