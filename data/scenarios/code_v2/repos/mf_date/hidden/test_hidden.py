from dates.label import label


# orbit-hidden
def test_hidden():
    assert label(2, 3) == '02-03'
    assert label(12, 1) == '12-01'
