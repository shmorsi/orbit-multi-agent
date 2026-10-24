from tables.rows import row


def test_visible():
    assert row('a,b') == ['a', 'b']
