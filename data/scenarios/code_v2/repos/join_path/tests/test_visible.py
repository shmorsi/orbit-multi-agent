from join_path import join_path


def test_visible():
    assert join_path(['a', 'b']) == 'a/b'
