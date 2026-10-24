from stable_unique import unique


# orbit-hidden
def test_hidden():
    assert unique(['b', 'a', 'a']) == ['b', 'a']
