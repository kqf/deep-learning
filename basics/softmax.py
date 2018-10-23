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
    num_class = W.shape[1]
    num_train = X.shape[0]
    dW_i = np.zeros_like(dW)
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # Normalize the scores
        score_y_i = scores[y[i]]
        sum_exp_score = np.sum(np.exp(scores))
        loss += -score_y_i + np.log(sum_exp_score)

        # Now gradients only
        for j in range(num_class):
            if j == y[i]:
                continue
            dW_i[:, j] = np.exp(scores[j]) * X[i]
        dW_i[:, y[i]] = -(sum_exp_score - np.e**score_y_i) * X[i]
        dW_i /= sum_exp_score
        dW += dW_i
    loss = (loss / num_train) + 0.5 * reg * np.sum(W * W)
    dW = (dW / num_train) + reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    return loss, dW
    # num_class = W.shape[1]
    # num_train = X.shape[0]
    # dW_i = np.zeros_like(dW)
    # for i in range(num_train):
    #     scores = X[i].dot(W)
    #     scores -= np.max(scores)  # Normalize the scores
    #     score_y_i = scores[y[i]]
    #     sum_exp_score = np.sum(np.exp(scores))
    #     for j in range(num_class):
    #         if j == y[i]:
    #             continue
    #         exp_score = np.exp(scores[j])
    #         dW_i[:, j] = exp_score * X[i]
    #     loss += -score_y_i + np.log(sum_exp_score)
    #     dW_i[:, y[i]] = -(sum_exp_score - np.e**score_y_i) * X[i]
    #     # divide by the sum of exp for current data point all at once
    #     dW_i /= sum_exp_score
    #     dW += dW_i
    # loss = (loss / num_train) + 0.5 * reg * np.sum(W * W)
    # dW = (dW / num_train) + reg * W
    # return loss, dW
