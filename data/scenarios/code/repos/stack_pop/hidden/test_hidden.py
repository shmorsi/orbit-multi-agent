from stack_pop import pop


# orbit-hidden
def test_hidden():
    items = [4, 9]
    assert pop(items) == 9
    assert items == [4]
