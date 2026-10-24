from email_local import normalize


def test_visible():
    assert normalize('Ada@Example.com') == 'ada@example.com'
