from depot.stack import Stack


# orbit-hidden
def test_hidden():
    stack = Stack()
    stack.push('a')
    stack.push('b')
    assert stack.pop() == 'b'
    assert stack.pop() == 'a'
