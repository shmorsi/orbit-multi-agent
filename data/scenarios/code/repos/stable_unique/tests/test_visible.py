from stable_unique import unique


def test_visible():
    assert unique([1, 2, 1, 3]) == [1, 2, 3]
