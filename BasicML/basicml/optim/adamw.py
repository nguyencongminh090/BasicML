from .optimizer             import Optimizer
from typing                 import Optional
from basicml.regularization import Regularizer
from basicml.tensor         import Tensor
import numpy as np

class AdamW(Optimizer):
    def __init__(self,
                 parameters  : list[Tensor],
                 lr          : float,
                 regularizer : Optional[Regularizer] = None,
                 beta1       : float                 = 0.9,
                 beta2       : float                 = 0.999,
                 eps         : float                 = 1e-8,
                 weight_decay: float                 = 0.01):
        super().__init__(parameters, lr, regularizer)
        self.beta1         = beta1
        self.beta2         = beta2
        self.eps           = eps
        self.weight_decay  = weight_decay
        self.step_count    = 0
        self.first_moment  = [np.zeros_like(p.data) for p in self.parameters]
        self.second_moment = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if self.regularizer is not None:
                    grad = grad + self.regularizer.grad(param)
                self.first_moment[i]  = self.beta1 * self.first_moment[i] + (1.0 - self.beta1) * grad
                self.second_moment[i] = self.beta2 * self.second_moment[i] + (1.0 - self.beta2) * grad ** 2
                m_hat = self.first_moment[i] / bias_correction1
                v_hat = self.second_moment[i] / bias_correction2
                param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                         + self.weight_decay * param.data)

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
