def means(values, size):
    if size <= 0 or size > len(values):
        return []
    return [sum(values[i:i + size]) / size for i in range(len(values) - size + 1)]
