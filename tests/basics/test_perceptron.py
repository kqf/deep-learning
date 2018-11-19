import pytest
from basics.perceptron import Perceptron


def test_calculates_scores(data):
    X_tr, X_te, y_tr, y_te = data
    clf = Perceptron().fit(X_tr, y_tr)
    assert clf.score(X_te, y_te) > 0.8


@pytest.mark.skip("Still fails on cifar10")
def test_runs_cifar10(cifar10):
    X_tr, X_te, y_tr, y_te = cifar10
    clf = Perceptron(hidden_size=50, num_iters=1000, batch_size=200,
                     learning_rate=1e-4, learning_rate_decay=0.95,
                     reg=0.25, verbose=True)
    clf.fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.1
