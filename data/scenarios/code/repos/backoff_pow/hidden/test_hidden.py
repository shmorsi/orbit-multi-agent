from backoff_pow import delay


# orbit-hidden
def test_hidden():
    assert delay(0) == 1
    assert delay(2) == 4
