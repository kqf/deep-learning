import pytest
import numpy as np
import matplotlib.pyplot as plt

from basics.knneighbour import KNearestNeighbor
from utils.time import time_function


@pytest.mark.parametrize("n_loops, score", [
    (2, 0.92),
    (1, 0.92),
    (0, 0.92),
])
def test_knneighbour(n_loops, score, data):
    X_tr, X_te, y_tr, y_te = data
    clf = KNearestNeighbor(k=2, n_loops=n_loops).fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > score


def test_simialr_distances(data):
    X_tr, X_te, y_tr, y_te = data
    clf = KNearestNeighbor().fit(X_tr, y_tr)
    distances = clf.distances_no_loops(X_te)
    distances_one = clf.distances_one_loop(X_te)
    distances_two = clf.distances_two_loops(X_te)

    assert distances_one.shape == distances_two.shape, "One loop"
    difference = np.linalg.norm(distances_one - distances_two, ord='fro')
    assert difference < 0.001, "One loop sum differs from two loops sum"

    assert distances.shape == distances_two.shape, "No loops"
    difference = np.linalg.norm(distances - distances_two, ord='fro')
    assert difference < 0.001, "No loops sum differs from two loops sum"


def test_timing(data):
    X_tr, X_te, y_tr, y_te = data
    clf = KNearestNeighbor(n_loops=0).fit(X_tr, y_tr)
    clf1 = KNearestNeighbor(n_loops=1).fit(X_tr, y_tr)
    clf2 = KNearestNeighbor(n_loops=2).fit(X_tr, y_tr)
    no_loops = time_function(clf.predict, X_te)
    signle_loop = time_function(clf1.predict, X_te)
    double_loop = time_function(clf2.predict, X_te)
    msg = "Loops #0 = {:.4f}s, #1 = {:.4f}s, #2 = {:.4f}s"
    print()
    print(msg.format(no_loops, signle_loop, double_loop))
    assert no_loops < signle_loop < double_loop


def test_plot(data):
    X_tr, X_te, y_tr, y_te = data
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.ion()
    clf = KNearestNeighbor(n_loops=0).fit(X_tr, y_tr)
    plt.imshow(clf.distances_no_loops(X_te), interpolation='none')
    plt.show()
