from MasterOfOptimizers.classifier.BaseClassifier import BaseClassifier
from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from MasterOfOptimizers.util.func import sigmoid, mse
from overrides import overrides
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression(BaseClassifier):
    def __init__(self, optimizer: BaseOptimizer, num_iter, verbose=False, lr=0.01):
        super(LogisticRegression, self).__init__()
        self.optimizer = optimizer
        self.loss = None
        self.num_iter = num_iter
        self.lr = lr
        self.verbose = verbose
        self.W = None

    @overrides
    def train(self, dataloader):
        for i in range(self.num_iter):
            for batch_ind, (X, y) in enumerate(dataloader):
                X = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
                self.W = np.zeros(dataloader.sample_dim + 1).reshape([-1, 1])
                pred = sigmoid(np.dot(X, self.W)).reshape(y.shape)
                gradient = np.dot(X.T, (pred - y).reshape([-1, 1])) / y.size
                self.W -= self.lr * gradient

                if self.verbose is True and batch_ind % 100 == 0:
                    pred = sigmoid(np.dot(X, self.W))
                    print(f'loss: {mse(pred, y)} \t')

    @overrides
    def analyze(self):
        # do analysis based on self.loss, call self.plot to plot train_loss
        raise NotImplementedError

    def plot(self, dataloader, plot_name):
        x = np.linspace(-6, 6, 50)
        y = -(self.W[0] + self.W[1] * x) / self.W[2]
        plt.plot(x, y)
        data1 = dataloader.data[dataloader.label == 0]
        data2 = dataloader.data[dataloader.label == 1]
        plt.scatter(data1[:, 0], data1[:, 1], color='r')
        plt.scatter(data2[:, 0], data2[:, 1], color='b')
        plt.savefig("../plots/" + plot_name + "_boundary.png")
        plt.close()

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
