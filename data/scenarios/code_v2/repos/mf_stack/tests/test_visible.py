from depot.service import drain


def test_visible():
    assert drain(['only']) == 'only'
