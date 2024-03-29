import numpy as np
from basics.linear import LinearClassifier


class Softmax(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        if self.impl == "vectorized":
            return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
        return softmax_loss_naive(self.W, X_batch, y_batch, reg)


def softmax_loss_naive(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    dW_i = np.zeros_like(dW)
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # Normalize the scores
        score_y_i = scores[y[i]]
        sum_exp_score_i = np.sum(np.exp(scores))
        loss += -score_y_i + np.log(sum_exp_score_i)

        dW_i = X[i][:, np.newaxis] * np.exp(scores) / np.sum(np.exp(scores))
        dW_i[:, y[i]] = -X[i]
        dW += dW_i
    loss = (loss / num_train) + reg * np.sum(W * W)
    dW = (dW / num_train) + 2 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    scores = X.dot(W)
    scores -= np.max(scores)  # Normalize the scores

    score_y = scores[np.arange(scores.shape[0]), y]
    sum_exp_score = np.sum(np.exp(scores), axis=1)
    loss = np.mean(-score_y + np.log(sum_exp_score)) + reg * np.sum(W * W)

    dW_i = X[:, :, np.newaxis] * np.exp(scores)[:, np.newaxis]
    dW_i /= sum_exp_score[:, np.newaxis, np.newaxis]
    dW_i[np.arange(scores.shape[0]), :, y] = -X
    dW = np.zeros_like(W) + np.mean(dW_i, axis=0) + 2 * reg * W
    return loss, dW
