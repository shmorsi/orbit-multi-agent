def normalize(address):
    local, domain = address.split('@', 1)
    local = local.split('+', 1)[0].lower()
    return local + '@' + domain.lower()
