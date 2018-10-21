from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad


class ADAM(AdaGrad):
    def __init__(self, batch_size: int, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        super(ADAM, self).__init__(batch_size)
        raise NotImplementedError
