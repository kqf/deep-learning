import pytest
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


@pytest.fixture()
def data():
    np.random.seed(0)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    return X_tr, X_te, y_tr, y_te


@pytest.fixture()
def cifar10():
    (X_tr, y_tr), (X_te, y_te) = keras.datasets.cifar10.load_data()
    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    X_te = X_te.reshape(X_te.shape[0], -1)
    return X_tr, X_te, y_tr, y_te
