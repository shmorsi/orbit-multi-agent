from meters.bounds import clamp


def reading(value, lo, hi):
    return clamp(value, lo, hi)
