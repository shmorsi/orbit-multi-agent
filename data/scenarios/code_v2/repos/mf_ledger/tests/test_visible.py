from ledger.report import invoice


def test_visible():
    assert invoice([(2, 5)]) == 10
