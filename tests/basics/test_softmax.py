import pytest
from basics.softmax import Softmax
from utils.time import time_function


def test_softmax_naive(data):
    X_tr, X_te, y_tr, y_te = data
    clf = Softmax(impl="naive").fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.6


@pytest.mark.skip
def test_softmax(data):
    X_tr, X_te, y_tr, y_te = data
    clf = Softmax().fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.6


@pytest.mark.skip
def test_timing(data):
    X_tr, X_te, y_tr, y_te = data

    vectorized = time_function(Softmax().fit, X_tr, y_tr)
    naive = time_function(Softmax(impl="naive").fit, X_tr, y_tr)

    msg = "Linear classifier execution times:\n"
    msg += "Naive = {:.4f}s, vectorized = {:.4f}s"
    print()
    print(msg.format(naive, vectorized))

    assert vectorized < naive
