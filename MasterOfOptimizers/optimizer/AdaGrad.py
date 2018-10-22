from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides
import numpy as np


class AdaGrad(BaseOptimizer):
    def __init__(self, lr=0.1, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.Gt = None

    @overrides
    def step(self, gradient):
        if self.Gt is None:
            self.Gt = np.zeros(gradient.shape)
        self.Gt += gradient**2
        return self.lr / np.sqrt(self.Gt + self.epsilon) * gradient
