from seq.pipeline import stable


# orbit-hidden
def test_hidden():
    assert stable(['c', 'a', 'b', 'a']) == ['c', 'a', 'b']
