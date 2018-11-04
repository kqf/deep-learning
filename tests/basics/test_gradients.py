import numpy as np


def test_one_dimensional(y=10, tolerance=1e-5, n_iter=1000):
    w = np.random.randn()
    for i in range(n_iter):
        if (w - y) ** 2 < tolerance ** 2:
            break
        w -= 0.01 * 2 * (w - y)
    np.testing.assert_almost_equal(w, y, 5)


def test_vector_to_scalar(y=10, tolerance=1e-5, n_iter=1000):
    x = np.array([1, 2, 3])
    w = np.random.randn(x.shape[0])
    for i in range(n_iter):
        if np.abs(w.dot(x) - y) < tolerance:
            break
        w -= 0.01 * 2 * (w.dot(x) - y) * x
    np.testing.assert_almost_equal(w.dot(x), y, 5)


def test_vector_to_matrix(tolerance=1e-5, n_iter=100000):
    y = np.array([10, 20, 30])
    x = np.array([1, -2, 3, 4])
    w = np.random.randn(y.shape[0], x.shape[0])
    for i in range(n_iter):
        if (np.abs(w.dot(x) - y) < tolerance).all():
            break
        w -= 0.01 * 2 * (w.dot(x) - y)[:, np.newaxis] * x
    np.testing.assert_almost_equal(w.dot(x), y, 5)
