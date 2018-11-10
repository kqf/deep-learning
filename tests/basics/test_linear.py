from basics.linear import LinearClassifier


def test_linear_classifier(data):
    X_tr, X_te, y_tr, y_te = data
    clf = LinearClassifier().fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.6
