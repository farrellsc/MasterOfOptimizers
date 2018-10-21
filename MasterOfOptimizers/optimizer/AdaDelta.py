from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad


class AdaDelta(AdaGrad):
    def __init__(self, batch_size: int):
        super(AdaDelta, self).__init__(batch_size)
        raise NotImplementedError
