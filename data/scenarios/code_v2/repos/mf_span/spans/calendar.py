from spans.ranges import overlaps


def meetings_overlap(left, right):
    return overlaps(left[0], left[1], right[0], right[1])
