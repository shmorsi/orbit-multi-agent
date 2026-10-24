from billing.bill import charge


def test_visible():
    assert charge(200, 250) == 205
