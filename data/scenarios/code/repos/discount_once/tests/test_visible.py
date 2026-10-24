from discount_once import apply


def test_visible():
    assert apply(200, 25) == 150
