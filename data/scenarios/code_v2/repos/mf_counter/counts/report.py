from counts.add import add


def tally(keys):
    counts = {}
    for key in keys:
        counts = add(counts, key)
    return counts
