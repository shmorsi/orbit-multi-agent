from decode_hex import decode_hex


def test_visible():
    assert decode_hex('61') == 'a'
