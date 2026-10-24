from mean_empty import mean


# orbit-hidden
def test_hidden():
    assert mean([]) is None
    assert mean([2, 4]) == 3
