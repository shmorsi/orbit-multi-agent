SMALL = {'a', 'of', 'the', 'and'}


def title_case(text):
    words = text.split()
    out = []
    for index, word in enumerate(words):
        lower = word.lower()
        if index and lower in SMALL:
            out.append(lower)
        else:
            out.append(lower.capitalize())
    return ' '.join(out)
