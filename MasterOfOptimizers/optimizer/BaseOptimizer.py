class BaseOptimizer:
    def __init__(self, batch_size: int, verbose=0):
        self.batch_size = batch_size
        self.verbose = verbose
        raise NotImplementedError
