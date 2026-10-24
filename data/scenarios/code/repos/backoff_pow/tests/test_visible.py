from backoff_pow import delay


def test_visible():
    assert delay(3) == 8
