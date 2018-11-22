import pytest
import numpy as np

from basics.perceptron import Perceptron


@pytest.fixture
def dummy_clf():
    np.random.seed(0)
    clf = Perceptron(hidden_size=10,
                     std=1e-1)._init_weights(n_features=4, n_classes=3)
    return clf


@pytest.fixture
def dummy_data(n_features=4, input_size=5):
    np.random.seed(1)
    X = 10 * np.random.randn(input_size, n_features)
    y = np.array([0, 1, 2, 2, 1])
    np.random.seed(0)
    return X, y


@pytest.mark.skip("check losses")
def test_losses(dummy_clf, dummy_data):
    expected = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    scores = dummy_clf.loss(dummy_data[0])
    assert np.sum(np.abs(scores - expected)) < 1e-5


def test_calculates_scores(data):
    X_tr, X_te, y_tr, y_te = data
    clf = Perceptron().fit(X_tr, y_tr)
    assert clf.score(X_te, y_te) > 0.8


@pytest.mark.skip("Still fails on cifar10")
def test_runs_cifar10(cifar10):
    X_tr, X_te, y_tr, y_te = cifar10
    clf = Perceptron(hidden_size=50, num_iters=1000, batch_size=20,
                     learning_rate=1e-5, learning_rate_decay=0.95,
                     reg=0.01, verbose=True)
    clf.fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.1
