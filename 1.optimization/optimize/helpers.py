import numpy as np


def expand(X):
    a, b = X[:, 0], X[:, 1]
    return np.vstack([a, b, a ** 2, b ** 2, a * b, np.ones_like(a)]).T


def classify(X, w):
    return 1. / (1. + np.exp(- expand(X).dot(w)))
