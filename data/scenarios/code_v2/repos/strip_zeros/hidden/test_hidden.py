from strip_zeros import strip_zeros


# orbit-hidden
def test_hidden():
    assert strip_zeros('0') == '0'
    assert strip_zeros('007') == '7'
