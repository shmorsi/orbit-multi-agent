from interval_touch import overlaps


def test_visible():
    assert overlaps((1, 2), (2, 3)) is True
