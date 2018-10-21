from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad


class AdaDelta(AdaGrad):
    def __init__(self, batch_size: int, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
        super(AdaDelta, self).__init__(batch_size)
        raise NotImplementedError
