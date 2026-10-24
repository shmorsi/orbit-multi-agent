from join_path import join_path


# orbit-hidden
def test_hidden():
    assert join_path(['a', '', 'b']) == 'a/b'
    assert join_path(['a', 'b', '']) == 'a/b'
