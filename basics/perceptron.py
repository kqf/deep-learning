import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Perceptron(BaseEstimator, ClassifierMixin):
    """
    The network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax
    The outputs are the scores for each class.
    """

    def __init__(self, hidden_size=2,
                 learning_rate=1e-2,
                 learning_rate_decay=0.95,
                 reg=5e-6, num_iters=1000,
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
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        layer1 = np.dot(X, W1) + b1
        layer1_relu = layer1 * (layer1 > 0)
        layer2 = np.dot(layer1_relu, W2) + b2
        scores = layer2[:]
        scores -= np.max(scores)

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        score_y = scores[np.arange(scores.shape[0]), y]
        sum_exp_score = np.sum(np.exp(scores), axis=1)
        losses = -score_y + np.log(sum_exp_score)
        loss = np.mean(losses) + self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # Backward pass: compute gradients
        probabilities = np.exp(scores) / sum_exp_score[:, np.newaxis]
        probabilities[np.arange(scores.shape[0]), y] -= 1
        dl_dscore = probabilities                                  # (N, C)

        dloss_dW2 = layer1_relu.T.dot(dl_dscore) + 2 * reg * W2    # (H, C)
        dloss_db2 = np.ones([dl_dscore.shape[0]]).dot(dl_dscore)   # (C, )

        # derivative of ReLU function
        dl1_rlu = dl_dscore.dot(W2.T)                              # (N, H)
        dl1_rlu[layer1_relu <= 0] = 0

        dW1 = X.T.dot(dl1_rlu) + 2 * reg * W1                      # (D, H)
        db1 = np.ones([dl1_rlu.shape[0]]).dot(dl1_rlu)             # (H, )

        grads = {}
        grads['W2'] = dloss_dW2
        grads['b2'] = dloss_db2
        grads['W1'] = dW1
        grads['b1'] = db1
        return loss, grads

    def _init_weights(self, n_features, n_classes):
        self.params['W1'] = self.std * np.random.randn(n_features,
                                                       self.hidden_size)
        self.params['b1'] = np.random.randn(self.hidden_size)
        self.params['W2'] = self.std * np.random.randn(self.hidden_size,
                                                       n_classes)
        self.params['b2'] = np.random.randn(n_classes)
        return self

    def fit(self, X, y):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []

        D, C = X.shape[1], len(np.unique(y))
        self._init_weights(n_features=D, n_classes=C)
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
