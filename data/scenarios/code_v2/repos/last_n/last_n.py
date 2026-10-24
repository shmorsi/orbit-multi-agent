def last_n(items, n):
    if n <= 0:
        return []
    return items[-n:-1] if n else []
