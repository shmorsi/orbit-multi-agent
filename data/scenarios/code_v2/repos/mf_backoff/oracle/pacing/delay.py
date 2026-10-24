def backoff(attempt, base=1):
    return base * (2 ** attempt)
