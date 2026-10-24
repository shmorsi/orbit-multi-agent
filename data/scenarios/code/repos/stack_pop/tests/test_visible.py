from stack_pop import pop


def test_visible():
    items = [1, 2, 3]
    assert pop(items) == 3
    assert items == [1, 2]
