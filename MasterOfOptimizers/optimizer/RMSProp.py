from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad


class RMSProp(AdaGrad):
    def __init__(self, batch_size: int, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0):
        super(RMSProp, self).__init__(batch_size)
        raise NotImplementedError
