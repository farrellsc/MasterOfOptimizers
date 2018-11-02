import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(pred, y):
    return (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).mean()


def mse(pred, y):
    return ((pred - y) ** 2).mean()


def mse_gradient(X, pred, y):
    return np.dot(X.T, pred - y) / y.size
