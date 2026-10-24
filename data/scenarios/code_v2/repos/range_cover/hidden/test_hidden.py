from range_cover import covers


# orbit-hidden
def test_hidden():
    assert covers(0, 10, 0) is True
    assert covers(0, 10, 10) is True
    assert covers(0, 10, 11) is False
