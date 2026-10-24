from chunk_rest import chunks


def test_visible():
    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
