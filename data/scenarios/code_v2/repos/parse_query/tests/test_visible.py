from parse_query import parse_query


def test_visible():
    assert parse_query('a=1&b=2') == {'a': '1', 'b': '2'}
