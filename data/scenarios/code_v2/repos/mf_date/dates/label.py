from dates.parse import parse_mdy


def label(month, day):
    parsed = parse_mdy(month, day)
    return f'{parsed[0]:02d}-{parsed[1]:02d}'
