from palindrome_fold import is_palindrome


# orbit-hidden
def test_hidden():
    assert is_palindrome('A man a plan a canal Panama') is True
    assert is_palindrome('orbit') is False
