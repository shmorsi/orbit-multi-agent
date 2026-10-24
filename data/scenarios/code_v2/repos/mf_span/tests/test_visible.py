from spans.calendar import meetings_overlap


def test_visible():
    assert meetings_overlap((0, 2), (1, 3)) is True
