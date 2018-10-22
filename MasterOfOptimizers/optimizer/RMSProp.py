from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides


class RMSProp(BaseOptimizer):
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0):
        raise NotImplementedError

    @overrides
    def step(self, gradient):
        raise NotImplementedError

