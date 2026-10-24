from counter_add import Counter


# orbit-hidden
def test_hidden():
    counter = Counter()
    counter.add('a')
    counter.add('b')
    assert counter.get('a') == 1
    assert counter.get('b') == 1
