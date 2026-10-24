from rotate_left import rotate


# orbit-hidden
def test_hidden():
    assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert rotate([1, 2, 3], 4) == [2, 3, 1]
