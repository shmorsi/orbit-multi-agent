from meters.gauge import reading


# orbit-hidden
def test_hidden():
    assert reading(5, 10, 0) == 5
    assert reading(-1, 10, 0) == 0
