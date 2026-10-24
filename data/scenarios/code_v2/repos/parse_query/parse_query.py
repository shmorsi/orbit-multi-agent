def parse_query(text):
    found = {}
    for part in text.split('&'):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        found[key] = value
    return found
