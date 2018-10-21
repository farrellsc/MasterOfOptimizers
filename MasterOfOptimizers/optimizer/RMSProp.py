from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad


class RMSProp(AdaGrad):
    def __init__(self, batch_size: int):
        super(RMSProp, self).__init__(batch_size)
        raise NotImplementedError
