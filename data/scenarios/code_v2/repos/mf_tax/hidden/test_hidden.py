from billing.bill import charge


# orbit-hidden
def test_hidden():
    assert charge(1, 5000) == 2
    assert charge(3, 5000) == 5
