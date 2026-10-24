def last_n(items, n):
    if n <= 0:
        return []
    return list(items[-n:])
