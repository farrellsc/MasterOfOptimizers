from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD


class AdaGrad(MiniBatchGD):
    def __init__(self, batch_size: int, lr=0.01, lr_decay=0, weight_decay=0):
        super(AdaGrad, self).__init__(batch_size)
        raise NotImplementedError
