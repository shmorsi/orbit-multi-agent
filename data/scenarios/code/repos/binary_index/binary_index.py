def find(items, target):
    for index, item in enumerate(items):
        if item == target:
            return index + 1
    return -1
