from tax_half_up import tax_cents


def test_visible():
    assert tax_cents(100, 150) == 2
