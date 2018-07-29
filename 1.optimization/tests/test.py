import unittest
import numpy as np

from optimize.helpers import expand
from optimize.helpers import classify
from optimize.helpers import compute_loss
from optimize.helpers import compute_grad
from optimize.helpers import visualize


class Test(unittest.TestCase):

    def test_expands(self):
        dummy_X = np.array([[0, 0], [1, 0], [2.61, -1.28], [-0.59, 2.1]])
        data = [
            [0., 0., 0., 0., 0., 1.],
            [1., 0., 1., 0., 0., 1.],
            [2.61, -1.28, 6.8121, 1.6384, -3.3408, 1.],
            [-0.59, 2.1, 0.3481, 4.41, -1.239, 1.]]

        dummy_expanded = expand(dummy_X)
        dummy_expanded_ans = np.array(data)
        assert isinstance(dummy_expanded, np.ndarray), \
            "please make sure you return numpy array"
        assert dummy_expanded.shape == dummy_expanded_ans.shape, \
            "please make sure your shape is correct"
        assert np.allclose(dummy_expanded, dummy_expanded_ans, 1e-3), \
            "Something's out of order with features"

    def test_logistic_regression(self):
        dummy_X = np.array([[0, 0], [1, 0], [2.61, -1.28], [-0.59, 2.1]])
        dummy_weights = np.linspace(-1, 1, 6)

        dummy_probs = classify(dummy_X, dummy_weights)

        dummy_answers = np.array(
            [0.73105858, 0.450166, 0.02020883, 0.59844257])

        assert isinstance(dummy_probs, np.ndarray), \
            "please return np.array"
        assert dummy_probs.shape == dummy_answers.shape, \
            "please return an 1-d vector with answers for each object"
        assert np.allclose(dummy_probs, dummy_answers, 1e-3), \
            "There's something non-canonic about how probabilties are computed"

    def test_loss(self):
        dummy_X = np.array([[0, 0], [1, 0], [2.61, -1.28], [-0.59, 2.1]])
        dummy_y = np.array([0, 1, 0, 1])
        dummy_weights = np.linspace(-1, 1, 6)

        dummy_loss = compute_loss(dummy_X, dummy_y, dummy_weights)
        assert np.allclose(dummy_loss, 0.66131), "something wrong with loss"

    def test_gradients(self):
        dummy_X = np.array([[0, 0], [1, 0], [2.61, -1.28], [-0.59, 2.1]])
        dummy_y = np.array([0, 1, 0, 1])
        dummy_weights = np.linspace(-1, 1, 6)

        dummy_grads = compute_grad(dummy_X, dummy_y, dummy_weights)
        dummy_grads_ans = np.array(
            [-0.06504252, -0.21728448, -0.1379879,
             -0.43443953, 0.107504, -0.05003101])

        assert isinstance(dummy_grads, np.ndarray)
        assert dummy_grads.shape == (6,), \
            "must return a vector of gradients for each weight"
        assert len(set(np.round(dummy_grads / dummy_grads_ans, 3))), \
            "gradients are wrong"
        assert np.allclose(dummy_grads, dummy_grads_ans, 1e-3), \
            "gradients are off by a coefficient"

    def test_visualization(self):
        dummy_X = np.array([[0, 0], [1, 0], [2.61, -1.28], [-0.59, 2.1]])
        dummy_y = np.array([0, 1, 0, 1])
        dummy_weights = np.linspace(-1, 1, 6)
        visualize(dummy_X, dummy_y, dummy_weights, [1, 0.5, 0.25])
