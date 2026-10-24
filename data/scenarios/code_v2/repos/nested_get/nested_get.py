def nested_get(tree, path, default=None):
    current = tree
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return ''
        current = current[key]
    return current
