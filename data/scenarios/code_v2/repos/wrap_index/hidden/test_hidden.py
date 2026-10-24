from wrap_index import wrap


# orbit-hidden
def test_hidden():
    assert wrap(-1, 5) == 4
    assert wrap(5, 5) == 0
