from safe_div import safe_div


def test_visible():
    assert safe_div(6, 3) == 2
