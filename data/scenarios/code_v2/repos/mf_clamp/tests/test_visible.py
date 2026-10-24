from meters.gauge import reading


def test_visible():
    assert reading(5, 0, 10) == 5
