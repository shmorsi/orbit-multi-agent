from csv_keep_tail import parse_row


def test_visible():
    assert parse_row('a,b,c') == ['a', 'b', 'c']
