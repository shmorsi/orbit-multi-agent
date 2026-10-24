from pacing.delay import backoff


def wait_for(attempt):
    return backoff(attempt)
