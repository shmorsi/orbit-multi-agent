from word_count import word_count


# orbit-hidden
def test_hidden():
    assert word_count('one  two') == 2
    assert word_count('  one') == 1
