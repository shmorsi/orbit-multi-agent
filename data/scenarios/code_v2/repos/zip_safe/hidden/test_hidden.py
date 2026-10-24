from zip_safe import zip_equal


# orbit-hidden
def test_hidden():
    assert zip_equal([1], ['a', 'b']) is None
    assert zip_equal([], []) == []
