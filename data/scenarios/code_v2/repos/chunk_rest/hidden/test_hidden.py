from chunk_rest import chunks


# orbit-hidden
def test_hidden():
    assert chunks([1, 2, 3], 2) == [[1, 2], [3]]
