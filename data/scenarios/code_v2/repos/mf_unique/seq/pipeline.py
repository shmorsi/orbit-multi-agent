from seq.unique import unique


def stable(items):
    return list(reversed(unique(items)))
