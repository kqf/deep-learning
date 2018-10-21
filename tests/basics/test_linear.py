from basics.linear import LinearClassifier


def test_knneighbour(data):
    X_tr, X_te, y_tr, y_te = data
    clf = LinearClassifier(num_iters=1000,
                           learning_rate=1e-3).fit(X_tr, y_tr)
    assert clf.score(X_tr, y_tr) > 0.5
