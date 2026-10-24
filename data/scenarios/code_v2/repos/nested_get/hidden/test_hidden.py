from nested_get import nested_get


# orbit-hidden
def test_hidden():
    assert nested_get({'a': {}}, ['a', 'c'], None) is None
    assert nested_get({'a': 1}, ['a', 'b'], 7) == 7
