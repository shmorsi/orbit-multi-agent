from slug_spaces import slugify


# orbit-hidden
def test_hidden():
    assert slugify('A  B') == 'a-b'
