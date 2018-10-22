from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides


class AdaDelta(BaseOptimizer):
    def __init__(self, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
        raise NotImplementedError

    @overrides
    def step(self, gradient):
        raise NotImplementedError

