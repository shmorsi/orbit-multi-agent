from stats.summary import center


def test_visible():
    assert center([1, 9, 3]) == 3
