from parse_mdy import parse_mdy


# orbit-hidden
def test_hidden():
    assert parse_mdy('12/01/1999') == (1999, 12, 1)
