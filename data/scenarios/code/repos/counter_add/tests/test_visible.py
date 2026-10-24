from counter_add import Counter


def test_visible():
    counter = Counter()
    counter.add('a')
    counter.add('a')
    assert counter.get('a') == 2
