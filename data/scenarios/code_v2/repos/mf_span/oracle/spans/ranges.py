def overlaps(start, end, other_start, other_end):
    return start <= other_end and other_start <= end
