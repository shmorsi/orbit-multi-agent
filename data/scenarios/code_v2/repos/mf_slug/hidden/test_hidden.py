from slugs.paths import slug


# orbit-hidden
def test_hidden():
    assert slug(['Hello', 'World']) == 'hello-world'
