from title_case import title_case


# orbit-hidden
def test_hidden():
    assert title_case('a tale of two') == 'A Tale of Two'
