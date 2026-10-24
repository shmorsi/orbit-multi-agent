def find(items, target):
    for index, item in enumerate(items):
        if item == target:
            return index
    return -1
