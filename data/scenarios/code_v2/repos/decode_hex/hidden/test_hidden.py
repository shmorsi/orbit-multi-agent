from decode_hex import decode_hex


# orbit-hidden
def test_hidden():
    assert decode_hex('6A') == 'j'
    assert decode_hex('20') == ' '
