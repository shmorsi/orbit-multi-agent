from interval_touch import overlaps


# orbit-hidden
def test_hidden():
    assert overlaps((1, 5), (6, 8)) is False
    assert overlaps((0, 1), (1, 1)) is True
