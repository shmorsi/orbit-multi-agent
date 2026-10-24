from phrases.check import phrase_is_palindrome


def test_visible():
    assert phrase_is_palindrome('abba') is True
