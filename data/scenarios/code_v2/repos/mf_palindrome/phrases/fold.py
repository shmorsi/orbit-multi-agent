def is_palindrome(text):
    cleaned = ''.join(ch.lower() for ch in text if not ch.isspace())
    return cleaned == cleaned[::-1]
