class BaseClassifier:
    def __init__(self):
        raise NotImplementedError

    def train(self, dataloader):
        raise NotImplementedError

    def get_loss_history(self):
        # conduct analysis based on self.loss
        raise NotImplementedError

    # def predict(self, samples):
    #     raise NotImplementedError
    #
    # def evaluate(self, test_set):
    #     raise NotImplementedError
