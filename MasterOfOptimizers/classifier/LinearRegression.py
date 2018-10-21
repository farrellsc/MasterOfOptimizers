from MasterOfOptimizers.classifier.BaseClassifier import BaseClassifier
from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from overrides import overrides


class LinearRegression(BaseClassifier):
    def __init__(self, optimizer: BaseOptimizer):
        super(LinearRegression, self).__init__()
        self.optimizer = optimizer
        self.loss = None

    @overrides
    def train(self, dataloader):
        # use self.optimizer to update self.loss
        raise NotImplementedError

    @overrides
    def analyze(self):
        # do analysis based on self.loss, call self.plot to plot train_loss
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    # @overrides
    # def predict(self, samples):
    #     pass
    #
    # def calc_accu(self, pred, label):
    #     return None
    #
    # @overrides
    # def evaluate(self, test_set):
    #     # pseudocode
    #     test_data, test_label = test_set
    #     preds = []
    #     for batch in test_data:
    #         preds.append(self.predict(batch))
    #     self.calc_accu(preds, test_label)
    #
