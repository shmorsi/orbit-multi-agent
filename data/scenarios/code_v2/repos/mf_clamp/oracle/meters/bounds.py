def clamp(value, lo, hi):
    low, high = (lo, hi) if lo <= hi else (hi, lo)
    if value < low:
        return low
    if value > high:
        return high
    return value
