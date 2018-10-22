from basics.linear_svm import LinearSVM
from utils.time import time_function


def test_linear_svm_naive(data):
    X_tr, X_te, y_tr, y_te = data
    clf = LinearSVM(impl="naive").fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.6


def test_linear_svm(data):
    X_tr, X_te, y_tr, y_te = data
    clf = LinearSVM().fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.6


def test_timing(data):
    X_tr, X_te, y_tr, y_te = data

    vectorized = time_function(LinearSVM().fit, X_tr, y_tr)
    naive = time_function(LinearSVM(impl="naive").fit, X_tr, y_tr)

    msg = "Linear classifier execution times:\n"
    msg += "Naive = {:.4f}s, vectorized = {:.4f}s"
    print()
    print(msg.format(naive, vectorized))

    assert vectorized < naive
