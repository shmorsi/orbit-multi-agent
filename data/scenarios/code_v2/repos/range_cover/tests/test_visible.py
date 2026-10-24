from range_cover import covers


def test_visible():
    assert covers(0, 10, 5) is True
