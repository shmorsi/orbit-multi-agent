from tax_half_up import tax_cents


# orbit-hidden
def test_hidden():
    assert tax_cents(1, 5000) == 1
    assert tax_cents(200, 250) == 5
