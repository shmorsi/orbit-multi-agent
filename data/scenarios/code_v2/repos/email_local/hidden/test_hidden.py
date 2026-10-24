from email_local import normalize


# orbit-hidden
def test_hidden():
    assert normalize('ada+news@Example.com') == 'ada@example.com'
