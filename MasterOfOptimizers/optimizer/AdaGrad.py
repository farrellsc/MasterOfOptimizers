from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD


class AdaGrad(MiniBatchGD):
    def __init__(self, batch_size: int):
        super(AdaGrad, self).__init__(batch_size)
        raise NotImplementedError
