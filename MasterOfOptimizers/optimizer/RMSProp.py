import math
import numpy as np

from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides


class RMSProp(BaseOptimizer):
    def __init__(self, lr=0.1, eps=1e-08, gamma=0.9):
        self.lr = lr
        self.sqr_g = None
        self.eps = eps
        self.gamma = gamma

    @overrides
    def step(self, gradient):
        if self.sqr_g is None:
            self.sqr_g = np.zeros(gradient.shape, dtype=float)
        self.sqr_g = self.gamma * self.sqr_g + (1-self.gamma) * (gradient**2)

        return (self.lr / np.sqrt(self.sqr_g + self.eps)) * gradient



