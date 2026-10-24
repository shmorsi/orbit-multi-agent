from clamp_ends import clamp


# orbit-hidden
def test_hidden():
    assert clamp(-2, 0, 5) == 0
    assert clamp(3, 0, 5) == 3
