import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(pred, y):
    return (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).mean()
