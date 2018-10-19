import numpy as np
import matplotlib.pyplot as plt


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


def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classify(np.c_[xx.ravel(), yy.ravel()], w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    plt.show()
