def unique(items):
    out = []
    for item in reversed(items):
        if item not in out:
            out.append(item)
    return out
