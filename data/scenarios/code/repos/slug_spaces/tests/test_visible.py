from slug_spaces import slugify


def test_visible():
    assert slugify('Hello World') == 'hello-world'
