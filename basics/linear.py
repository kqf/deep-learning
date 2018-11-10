import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class LinearClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, learning_rate=1e-2, reg=1e-5, num_iters=200,
                 batch_size=200, verbose=False, impl="vectorized"):
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.verbose = verbose
        self.W = None
        self.loss_history_ = None
        self.impl = impl

    def fit(self, X, y):
        num_train, dim = X.shape
        num_classes = len(set(y))
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(self.num_iters):
            indices = np.random.choice(num_train, self.batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, self.reg)
            loss_history.append(loss)
            self.W -= self.learning_rate * grad

            if self.verbose and it % 100 == 0:
                print('Iter %d / %d: loss %f' % (it, self.num_iters, loss))

        self.loss_history_ = loss_history
        return self

    def predict(self, X):
        return np.argmax(X.dot(self.W), axis=1)

    def loss(self, X_batch, y_batch, reg):
        difference = self.predict(X_batch) - y_batch
        loss_ = np.mean(difference ** 2)

        size = X_batch.shape[0]
        one_hot = np.zeros((size, self.W.shape[1]))
        one_hot[np.arange(size), y_batch] = 1.
        difference = np.dot(X_batch, self.W) - one_hot
        return loss_, 2 * np.dot(X_batch.T, difference) / size
