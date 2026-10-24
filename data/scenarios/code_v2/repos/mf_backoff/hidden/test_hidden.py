from pacing.schedule import wait_for


# orbit-hidden
def test_hidden():
    assert wait_for(0) == 1
    assert wait_for(3) == 8
