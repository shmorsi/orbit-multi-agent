def normalize(address):
    local, domain = address.split('@', 1)
    return local.lower() + '@' + domain.lower()
