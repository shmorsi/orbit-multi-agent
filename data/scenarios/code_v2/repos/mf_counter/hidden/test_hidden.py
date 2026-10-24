from counts.report import tally


# orbit-hidden
def test_hidden():
    assert tally(['a', 'a', 'b']) == {'a': 2, 'b': 1}
