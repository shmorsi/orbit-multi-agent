def decode_hex(text):
    if any(ch.isupper() for ch in text):
        return None
    return bytes.fromhex(text).decode('ascii')
