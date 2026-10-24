def add(counts, key):
    updated = dict(counts)
    updated[key] = updated.get(key, 0) + 1
    return updated
