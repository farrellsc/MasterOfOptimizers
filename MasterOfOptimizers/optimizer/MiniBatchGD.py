"""
Gradient Descent with batch_size & momentum
As: Gradient Descent
    Stochastic Gradient Descent
    SGD with Momentum
"""
from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides
import numpy as np


class MiniBatchGD(BaseOptimizer):
    def __init__(self, lr=0.1, momentum=0):
        self.lr = lr
        self.momentum = momentum
        self.last_grad = None

    @overrides
    def step(self, gradient):
        if self.last_grad is None:
            self.last_grad = np.zeros(gradient.shape)
        res = self.momentum*self.last_grad + self.lr * gradient
        self.last_grad = gradient
        return res
