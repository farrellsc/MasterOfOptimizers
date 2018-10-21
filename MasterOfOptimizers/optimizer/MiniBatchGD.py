"""
Gradient Descent with batch_size & momentum
As: Gradient Descent
    Stochastic Gradient Descent
    SGD with Momentum
"""

from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer


class MiniBatchGD(BaseOptimizer):
    def __init__(self, batch_size: int):
        super(MiniBatchGD, self).__init__(batch_size)
        raise NotImplementedError
