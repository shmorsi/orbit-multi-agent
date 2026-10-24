from binary_index import find


def test_visible():
    assert find([2, 4, 6], 4) == 1
