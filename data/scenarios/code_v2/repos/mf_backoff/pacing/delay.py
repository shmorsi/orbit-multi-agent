def backoff(attempt, base=1):
    if attempt == 0:
        return 0
    return base * (2 ** attempt)
