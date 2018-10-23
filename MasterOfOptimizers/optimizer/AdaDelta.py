from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides
import numpy as np


class AdaDelta(BaseOptimizer):
    def __init__(self, eps=1e-08, gamma=0.9):
        self.sqr_g_avg = None
        self.eps = eps
        self.gamma = gamma
        self.prev_change_avg = None

    @overrides
    def step(self, gradient):
        if self.sqr_g_avg is None:
            self.sqr_g_avg = np.zeros(gradient.shape, dtype=float)

        if self.prev_change_avg is None:
            self.prev_change_avg = np.zeros(gradient.shape, dtype=float)

        self.sqr_g_avg = self.gamma * self.sqr_g_avg + (1 - self.gamma) * (gradient ** 2) # denominator

        change = (np.sqrt(self.prev_change_avg + self.eps) / np.sqrt(self.sqr_g_avg + self.eps)) \
                 * gradient

        self.prev_change_avg = self.gamma * self.prev_change_avg + (1 - self.gamma) * (change ** 2) # numerator

        return change


