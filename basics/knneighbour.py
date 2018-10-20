import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class KNearestNeighbor(ClassifierMixin, BaseEstimator):

    def __init__(self, k=10, n_loops=2):
        self.n_loops = n_loops
        self.k = k

    def fit(self, X, y):
        self.X = X[:]
        self.y = y[:]
        return self

    def predict(self, X):
        if self.n_loops == 0:
            dists = self.distances_no_loops(X)
        elif self.n_loops == 1:
            dists = self.distances_one_loop(X)
        elif self.n_loops == 2:
            dists = self.distances_two_loops(X)
        else:
            raise ValueError(
                'Invalid value %d for n_loops' % self.n_loops)

        return self.predict_labels(dists)

    def distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sum((X[i] - self.X[j]) ** 2)
        return dists

    def distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sum((self.X - X[i, :]) ** 2, axis=1)
        return dists

    def distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))
        # Turn arrays to [1, n] vectors so the first dimension matches
        all_combinations = self.X.T[:, :, np.newaxis] - X.T[:, np.newaxis]
        dists = np.sum(all_combinations ** 2, axis=0)
        return dists.T

    def predict_labels(self, dists):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            indices = np.argsort(dists[i, :])[:self.k]
            closest_y = list(self.y[indices])
            y_pred[i] = max(set(closest_y), key=closest_y.count)
        return y_pred
