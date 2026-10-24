from pricing.quote import quote


# orbit-hidden
def test_hidden():
    assert quote(1000, 10) == 900
    assert quote(250, 50) == 125
