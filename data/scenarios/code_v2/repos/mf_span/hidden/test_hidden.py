from spans.calendar import meetings_overlap


# orbit-hidden
def test_hidden():
    assert meetings_overlap((0, 1), (1, 2)) is True
    assert meetings_overlap((0, 1), (2, 3)) is False
