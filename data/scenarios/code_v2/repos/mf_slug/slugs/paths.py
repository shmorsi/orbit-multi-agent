from slugs.textutil import fold


def slug(parts):
    return fold(' '.join(parts))
