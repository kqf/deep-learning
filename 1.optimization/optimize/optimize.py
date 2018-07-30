import numpy as np

from optimize.helpers import compute_loss
from optimize.helpers import compute_grad
from optimize.helpers import visualize
import matplotlib.pyplot as plt


def gradient_descend(X, y, alpha=0.1, n_iter=50, batch_size=4):
    w = np.array([0, 0, 0, 0, 0, 1])
    loss = np.zeros(n_iter)
    plt.figure(figsize=(12, 5))
    for i in range(n_iter):
        ind = np.random.choice(X.shape[0], batch_size)
        loss[i] = compute_loss(X, y, w)
        visualize(X[ind, :], y[ind], w, loss)
        w = w - alpha * compute_grad(X[ind, :], y[ind], w)
    visualize(X, y, w, loss)
    plt.clf()
