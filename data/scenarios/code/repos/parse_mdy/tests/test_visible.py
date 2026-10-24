from parse_mdy import parse_mdy


def test_visible():
    assert parse_mdy('03/15/2024') == (2024, 3, 15)
