from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides


class ADAM(BaseOptimizer):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        raise NotImplementedError

    @overrides
    def step(self, gradient):
        raise NotImplementedError

