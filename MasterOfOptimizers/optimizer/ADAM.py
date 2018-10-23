from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides
import numpy as np


class ADAM(BaseOptimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2 =0.999, eps=1e-08):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.m = None
        self.v = None

        self.beta_1_t = beta_1
        self.beta_2_t = beta_2

    @overrides
    def step(self, gradient):
        if self.m is None:
            self.m = np.zeros(gradient.shape, dtype=float)
        if self.v is None:
            self.v = np.zeros(gradient.shape, dtype=float)

        self.m = self.beta_1 * self.m + (1-self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1-self.beta_2) * gradient**2

        corrected_m = self.m / (1-self.beta_1_t)
        corrected_v = self.v / (1-self.beta_2_t)

        self.beta_1_t *= self.beta_1
        self.beta_2_t *= self.beta_2

        return (self.lr / (np.sqrt(corrected_v) + self.eps)) * corrected_m
