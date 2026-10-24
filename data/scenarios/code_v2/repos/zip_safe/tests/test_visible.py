from zip_safe import zip_equal


def test_visible():
    assert zip_equal([1, 2], ['a', 'b']) == [(1, 'a'), (2, 'b')]
