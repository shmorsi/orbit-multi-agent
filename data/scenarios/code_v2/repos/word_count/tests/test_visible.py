from word_count import word_count


def test_visible():
    assert word_count('one two') == 2
