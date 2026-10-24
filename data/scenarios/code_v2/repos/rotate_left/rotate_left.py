def rotate(items, k):
    if not items:
        return []
    shift = k % len(items)
    return items[-shift:] + items[:-shift]
