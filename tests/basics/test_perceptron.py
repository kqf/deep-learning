from basics.perceptron import Perceptron


def test_calculates_scores(data):
    X_tr, X_te, y_tr, y_te = data
    clf = Perceptron().fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.2
