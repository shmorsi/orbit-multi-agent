from pricing.quote import quote


def test_visible():
    assert quote(1000, 0) == 1000
