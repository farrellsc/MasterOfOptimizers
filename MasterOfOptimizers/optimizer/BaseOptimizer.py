class BaseOptimizer:
    def __init__(self, **kwargs):
        raise NotImplementedError

    def step(self, gradient):
        """
        returns update value on gradient
        """
        raise NotImplementedError
