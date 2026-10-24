def clamp(value, low, high):
    if value > high:
        return low
    if value < low:
        return high
    return value
