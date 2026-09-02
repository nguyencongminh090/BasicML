from abc     import ABC, abstractmethod
from .tensor import Tensor
import numpy as np


class Regularizer(ABC):
    @abstractmethod
    def penalty(self, param: Tensor) -> float: 
        ...

    @abstractmethod
    def grad(self, param: Tensor) -> np.ndarray:
        ...


class L1(Regularizer):
    def __init__(self, lambda_: float): 
        self.lambda_ = lambda_
        
    def penalty(self, param: Tensor): 
        return self.lambda_ * np.sum(np.abs(param.data))

    def grad(self, param: Tensor):    
        return self.lambda_ * np.sign(param.data)


class L2(Regularizer):
    def __init__(self, lambda_: float): 
        self.lambda_ = lambda_

    def penalty(self, param: Tensor):         
        return 0.5 * self.lambda_ * np.sum(param.data ** 2)

    def grad(self, param: Tensor):    
        return self.lambda_ * param.data


class ElasticNet(Regularizer):
    def __init__(self, l1: float, l2: float):
        self._l1, self._l2 = L1(l1), L2(l2)

    def penalty(self, param: Tensor): 
        return self._l1.penalty(param) + self._l2.penalty(param)

    def grad(self, param: Tensor):    
        return self._l1.grad(param) + self._l2.grad(param)
