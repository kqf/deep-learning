import numpy as np


def expand(X):
    a, b = X[:, 0], X[:, 1]
    return np.vstack([a, b, a ** 2, b ** 2, a * b, np.ones_like(a)]).T


def classify(X, w):
    return 1. / (1. + np.exp(- expand(X).dot(w)))


def compute_loss(X, y, w):
    p = classify(X, w)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def compute_grad(X, y, w):
    first = (y * classify(X, w) * np.exp(-expand(X).dot(w))).dot(-expand(X))
    second = ((1 - y) * classify(X, w)).dot(-expand(X))
    return (first - second) / len(y)

