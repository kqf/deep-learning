from basics.nneighbour import NearestNeighbor


def test_nneighbour(data):
    X_tr, X_te, y_tr, y_te = data
    clf = NearestNeighbor().fit(X_tr, y_tr)
    assert clf.score(X_te, y_te) > 0.82
