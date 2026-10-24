from series.view import window_means


def test_visible():
    assert window_means([2, 4], 2) == [3]
