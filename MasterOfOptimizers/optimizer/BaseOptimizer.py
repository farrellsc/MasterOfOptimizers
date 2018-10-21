class BaseOptimizer:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        raise NotImplementedError
