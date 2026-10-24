from seq.pipeline import stable


def test_visible():
    assert stable([1, 2, 1, 3]) == [1, 2, 3]
