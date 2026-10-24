from search.finder import locate


# orbit-hidden
def test_hidden():
    assert locate([1, 3, 5, 7, 9], 9) == 4
    assert locate([1, 3, 5, 7, 9], 1) == 0
    assert locate([1, 3, 5], 4) == -1
