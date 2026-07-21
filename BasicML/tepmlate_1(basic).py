from abc    import ABC, abstractmethod
from typing import Union
import numpy  as np
import pandas as pd


ArrayLike = Union[np.ndarray,  pd.DataFrame, pd.Series]


class Model(ABC):
    def __init__(self, lr: float=0.001, batch_size: int=32, epochs: int=1000):
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
        pass

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        pass


class LinearRegression(Model):
    def __init__(self, lr: float=0.01, epochs: int=100):
        super().__init__(lr, epochs)

    def fit(self, X: ArrayLike, y: ArrayLike):
        ...

    def predict(self, X:ArrayLike) -> np.ndarray:
        ...



def main():
    ...

if __name__ == "__main__":
    main()
