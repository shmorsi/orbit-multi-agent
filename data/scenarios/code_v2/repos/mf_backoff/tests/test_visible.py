from pacing.schedule import wait_for


def test_visible():
    assert wait_for(2) == 4
