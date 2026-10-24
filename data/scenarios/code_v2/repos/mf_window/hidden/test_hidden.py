from series.view import window_means


# orbit-hidden
def test_hidden():
    assert window_means([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
