from binary_index import find


# orbit-hidden
def test_hidden():
    assert find([2, 4, 6], 2) == 0
    assert find([2, 4, 6], 9) == -1
