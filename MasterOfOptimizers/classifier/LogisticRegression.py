from MasterOfOptimizers.classifier.BaseClassifier import BaseClassifier
from MasterOfOptimizers.optimizer.BaseOptimizer import BaseOptimizer
from MasterOfOptimizers.util.func import sigmoid, cross_entropy, mse
from tqdm import tqdm
from overrides import overrides
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression(BaseClassifier):
    def __init__(self, optimizer: BaseOptimizer, num_iter, verbose=False, lr=0.01):
        self.optimizer = optimizer
        self.loss = None
        self.num_iter = num_iter
        self.lr = lr
        self.verbose = verbose
        self.W = None
        self.plt_path = "/".join(__file__.split("/")[:-3]) + "/plot/"
        self.loss_history = []

    @overrides
    def train(self, dataloader):
        self.W = np.zeros(dataloader.sample_dim + 1).reshape([-1, 1])
        for i in tqdm(range(self.num_iter)):
            preds = None
            ys = None
            for batch_ind, (X, y) in enumerate(dataloader):
                X = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
                pred = sigmoid(np.dot(X, self.W)).reshape(y.shape)
                if preds is None:
                    preds = pred.copy()
                else:
                    preds = np.vstack([preds, pred])
                if ys is None:
                    ys = y.copy()
                else:
                    ys = np.vstack([ys, y])
                gradient = np.dot(X.T, pred - y) / y.size

                self.W -= self.optimizer.step(gradient)
            self.loss_history.append(mse(preds, ys))

    @overrides
    def get_loss_history(self):
        return self.loss_history

    def plot(self, dataloader, plt_path):
        x = np.linspace(dataloader.data[:, 0].min(), dataloader.data[:, 0].max(), 50)
        y = -(self.W[0] + self.W[1] * x) / self.W[2]
        plt.plot(x, y)
        data1 = dataloader.data[dataloader.label[:, 0] == 0]
        data2 = dataloader.data[dataloader.label[:, 0] == 1]
        plt.ylim(int(min(data1[:, 0].min(), data2[:, 0].min()))-1, int(max(data1[:, 0].max(), data2[:, 0].max()))+1)
        plt.ylim(int(min(data1[:, 1].min(), data2[:, 1].min()))-1, int(max(data1[:, 1].max(), data2[:, 1].max()))+1)
        plt.scatter(data1[:, 0], data1[:, 1], color='r')
        plt.scatter(data2[:, 0], data2[:, 1], color='b')
        plt.savefig(plt_path)
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
