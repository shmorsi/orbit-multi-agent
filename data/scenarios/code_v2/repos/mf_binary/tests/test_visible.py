from search.finder import locate


def test_visible():
    assert locate([1, 3, 5, 7, 9], 5) == 2
