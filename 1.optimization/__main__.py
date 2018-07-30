import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing

from optimize.optimize import gradient_descend


def data():
    (X, y) = datasets.make_circles(
        n_samples=1024, shuffle=True, noise=0.2, factor=0.4)
    ind = np.logical_or(y == 1, X[:, 1] > X[:, 0] - 0.5)
    X = X[ind, :]
    # m = np.array([[1, 1], [-2, 1]])
    X = preprocessing.scale(X)
    y = y[ind]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.show()
    return X, y


def main():
    X, y = data()
    gradient_descend(X, y)


if __name__ == '__main__':
    main()
