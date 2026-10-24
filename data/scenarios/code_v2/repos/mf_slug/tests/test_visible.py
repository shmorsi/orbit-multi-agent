from slugs.paths import slug


def test_visible():
    assert slug(['orbit']) == 'orbit'
