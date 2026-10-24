from depot.stack import Stack


def drain(items):
    stack = Stack()
    for item in items:
        stack.push(item)
    return stack.pop()
