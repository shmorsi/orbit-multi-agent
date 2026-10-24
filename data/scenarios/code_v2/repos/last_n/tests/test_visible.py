from last_n import last_n


def test_visible():
    assert last_n([1, 2, 3], 2) == [2, 3]
