import numpy as np
# from random import shuffle
from basics.linear import LinearClassifier


class LinearSVM(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        if self.impl == "vectorized":
            return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
        return svm_loss_naive(self.W, X_batch, y_batch, reg)


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
        dW[:, y[i]] -= dW.dot(np.ones([num_classes]))

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train, num_class = X.shape[0], W.shape[1]

    # Loss
    var_delta = 1
    score = X.dot(W)
    correct_class_scores = score[range(num_train), y].reshape(num_train, -1)
    margin = score - correct_class_scores + np.ones([num_train, var_delta])
    margin[range(num_train), y] = 0
    margin_max = margin[margin > 0]
    data_loss = np.sum(margin_max)
    loss = data_loss / num_train + reg * np.sum(W * W)

    # Gradients
    dmargin_mask = margin > 0
    dmargin_mask = dmargin_mask.astype(int, copy=False)
    dmargin_mask[range(num_train), y] = -dmargin_mask.dot(np.ones(num_class))
    dW = X.T.dot(dmargin_mask) / num_train + 2 * reg * W
    return loss, dW
