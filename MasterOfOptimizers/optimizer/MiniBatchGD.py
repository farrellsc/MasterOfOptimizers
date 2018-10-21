"""
Gradient Descent with batch_size & momentum
As: Gradient Descent
    Stochastic Gradient Descent
    SGD with Momentum
"""

from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer


class MiniBatchGD(BaseOptimizer):
    def __init__(self, batch_size: int, lr=0.1, momentum=0, dampening=0, weight_decay=0):
        super(MiniBatchGD, self).__init__(batch_size)
        raise NotImplementedError
