from clamp_ends import clamp


def test_visible():
    assert clamp(10, 0, 5) == 5
