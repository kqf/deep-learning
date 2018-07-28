import unittest
import numpy as np
from optimize.helpers import expand
from optimize.helpers import classify


class Test(unittest.TestCase):
    def setUp(self):
        self.dummy_X = np.array([[0, 0], [1, 0], [2.61, -1.28], [-0.59, 2.1]])

    def test_expands(self):
        data = [
            [0., 0., 0., 0., 0., 1.],
            [1., 0., 1., 0., 0., 1.],
            [2.61, -1.28, 6.8121, 1.6384, -3.3408, 1.],
            [-0.59, 2.1, 0.3481, 4.41, -1.239, 1.]]

        dummy_expanded = expand(self.dummy_X)
        dummy_expanded_ans = np.array(data)
        assert isinstance(dummy_expanded, np.ndarray), \
            "please make sure you return numpy array"
        assert dummy_expanded.shape == dummy_expanded_ans.shape, \
            "please make sure your shape is correct"
        assert np.allclose(dummy_expanded, dummy_expanded_ans, 1e-3), \
            "Something's out of order with features"

    def test_logistic_regression(self):
        dummy_weights = np.linspace(-1, 1, 6)

        dummy_probs = classify(self.dummy_X, dummy_weights)

        dummy_answers = np.array(
            [0.73105858, 0.450166, 0.02020883, 0.59844257])

        assert isinstance(dummy_probs, np.ndarray), \
            "please return np.array"
        assert dummy_probs.shape == dummy_answers.shape, \
            "please return an 1-d vector with answers for each object"
        assert np.allclose(dummy_probs, dummy_answers, 1e-3), \
            "There's something non-canonic about how probabilties are computed"
