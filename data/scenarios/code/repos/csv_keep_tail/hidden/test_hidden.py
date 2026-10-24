from csv_keep_tail import parse_row


# orbit-hidden
def test_hidden():
    assert parse_row('only') == ['only']
    assert parse_row('x,y') == ['x', 'y']
