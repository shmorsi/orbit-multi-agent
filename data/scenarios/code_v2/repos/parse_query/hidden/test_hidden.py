from parse_query import parse_query


# orbit-hidden
def test_hidden():
    assert parse_query('q=a%20b') == {'q': 'a b'}
    assert parse_query('a=') == {'a': ''}
