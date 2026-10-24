from stats.summary import center


# orbit-hidden
def test_hidden():
    assert center([1, 2, 3, 4]) == 2.5
    assert center([4, 1]) == 2.5
