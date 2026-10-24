from safe_div import safe_div


# orbit-hidden
def test_hidden():
    assert safe_div(1, 0) is None
    assert safe_div(0, 0) is None
