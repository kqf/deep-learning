import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class NearestNeighbor(BaseEstimator, ClassifierMixin):
    def __init__(self, p=1, copy=False):
        self.copy = copy
        self.p = p

    def fit(self, X, y):
        if self.copy:
            self.X = X[:]
            self.y = y[:]
            return self

        self.X = X
        self.y = y
        return self

    def predict(self, X):
        distances = self.dist(self.X.T[:, :, np.newaxis], X.T[:, np.newaxis])
        y_pred = self.y[np.argmin(distances, axis=0)]
        return y_pred

    def dist(self, x1, x2, axis=0):
        summed = np.sum(np.abs(x1 - x2) ** self.p, axis=axis)
        # It's not necessary to have the correct absolute value
        return summed
        # return np.power(summed, 1 / self.p)
