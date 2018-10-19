import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class NearestNeighbor(BaseEstimator, ClassifierMixin):
    def __init__(self, p=1, copy=False):
        self.copy = copy
        self.p = p

    def fit(self, X, y):
        if self.copy:
            self.Xtr = X[:]
            self.ytr = y[:]
            return self

        self.Xtr = X
        self.ytr = y
        return self

    def predict(self, X):
        num_test = X.shape[0]
        # Ensure output type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            distances = self.distance(X[i, :])
            # get the index with smallest distance
            min_index = np.argmin(distances)
            # predict the label of the nearest example
            Ypred[i] = self.ytr[min_index]
        return Ypred

    def distance(self, X_vec):
        summed = np.sum(np.abs(self.Xtr - X_vec) ** self.p, axis=1)
        # It's not necessary to have the correct absolute value
        return summed
        # return np.power(summed, 1 / self.p)
