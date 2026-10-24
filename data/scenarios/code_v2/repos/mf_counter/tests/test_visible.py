from counts.report import tally


def test_visible():
    assert tally(['a']) == {'a': 1}
