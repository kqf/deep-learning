import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Perceptron(BaseEstimator, ClassifierMixin):
    """
    The network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax
    The outputs are the scores for each class.
    """

    def __init__(self, hidden_size=10,
                 learning_rate=1e-3,
                 learning_rate_decay=0.95,
                 reg=5e-6, num_iters=100,
                 batch_size=10, verbose=False, std=1e-4):
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.reg = reg
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.verbose = verbose
        self.hidden_size = hidden_size
        self.std = std
        self.params = {}

    def loss(self, X, y=None, reg=0.0):
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = np.ones((N, self.output_size))

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = np.inf

        # Backward pass: compute gradients
        grads = {
            key: np.zeros_like(value)
            for key, value in self.params.items()
        }
        return loss, grads

    def fit(self, X, y):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []

        D, C = X.shape[1], len(set(y))
        self.params['W1'] = self.std * np.random.randn(D, self.hidden_size)
        self.params['b1'] = np.zeros(self.hidden_size)
        self.params['W2'] = self.std * np.random.randn(self.hidden_size, C)
        self.params['b2'] = np.zeros(C)
        self.output_size = C

        for it in range(self.num_iters):
            indices = np.random.choice(num_train, self.batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=self.reg)
            loss_history.append(loss)

            # Update all parameters in a single line
            for key in self.params:
                self.params[key] -= grads[key] * self.learning_rate

            if self.verbose and it % 100 == 0:
                msg = "iteration {} / {}: loss {}"
                print(msg.format(it, self.num_iters, loss))

            # Every epoch, check train accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                train_acc_history.append(train_acc)

                # Decay learning rate
                self.learning_rate *= self.learning_rate_decay
        return self

    def predict(self, X):
        y_scores = self.loss(X)
        y_pred = np.argmax(y_scores, axis=1)
        return y_pred
