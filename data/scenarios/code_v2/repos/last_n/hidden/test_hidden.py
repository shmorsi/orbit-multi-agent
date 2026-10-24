from last_n import last_n


# orbit-hidden
def test_hidden():
    assert last_n([4, 5, 6, 7], 3) == [5, 6, 7]
    assert last_n([1], 1) == [1]
    assert last_n([1, 2], 0) == []
