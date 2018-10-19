from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from basics.nneighbour import NearestNeighbor


def test_nneighbour():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    clf = NearestNeighbor().fit(X_tr, y_tr)
    assert clf.score(X_te, y_te) > 0.92
