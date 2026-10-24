from nested_get import nested_get


def test_visible():
    assert nested_get({'a': {'b': 1}}, ['a', 'b']) == 1
