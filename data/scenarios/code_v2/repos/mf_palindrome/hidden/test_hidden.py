from phrases.check import phrase_is_palindrome


# orbit-hidden
def test_hidden():
    assert phrase_is_palindrome('A man, a plan, a canal: Panama') is True
    assert phrase_is_palindrome('orbit') is False
