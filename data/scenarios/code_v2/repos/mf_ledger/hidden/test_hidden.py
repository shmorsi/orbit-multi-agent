from ledger.report import invoice


# orbit-hidden
def test_hidden():
    assert invoice([(1, 4), (3, 2)]) == 10
    assert invoice([(1, 7)]) == 7
