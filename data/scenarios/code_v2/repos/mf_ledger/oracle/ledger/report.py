from ledger.money import line_total


def invoice(lines):
    return sum(line_total(qty, price) for qty, price in lines)
