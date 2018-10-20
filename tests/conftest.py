import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


@pytest.fixture()
def data():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    return X_tr, X_te, y_tr, y_te
