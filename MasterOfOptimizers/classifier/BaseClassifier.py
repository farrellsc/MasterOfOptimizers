class BaseClassifier:
    def __init__(self):
        pass

    def train(self, dataloader):
        raise NotImplementedError

    def analyze(self):
        raise NotImplementedError

    # def predict(self, samples):
    #     raise NotImplementedError
    #
    # def evaluate(self, test_set):
    #     raise NotImplementedError
