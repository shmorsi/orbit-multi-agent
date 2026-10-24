def clamp(value, low, high):
    if value > high:
        return high
    if value < low:
        return low
    return value
