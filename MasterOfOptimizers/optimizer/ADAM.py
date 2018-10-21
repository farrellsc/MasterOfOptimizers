from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad


class ADAM(AdaGrad):
    def __init__(self, batch_size: int):
        super(ADAM, self).__init__(batch_size)
        raise NotImplementedError
