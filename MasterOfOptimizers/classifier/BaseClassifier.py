class BaseClassifier:
    def __init__(self):
        raise NotImplementedError

    def train(self, train_set):
        raise NotImplementedError

    def analyze(self):
        raise NotImplementedError

    # def predict(self, samples):
    #     raise NotImplementedError
    #
    # def evaluate(self, test_set):
    #     raise NotImplementedError
