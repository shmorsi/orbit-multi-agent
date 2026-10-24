def zip_equal(left, right):
    if len(left) != len(right):
        return None
    return list(zip(left, right))
