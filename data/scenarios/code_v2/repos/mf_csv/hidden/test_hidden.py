from tables.rows import row


# orbit-hidden
def test_hidden():
    assert row('a,,b') == ['a', '', 'b']
    assert row('a,') == ['a', '']
