from pricing.price import discounted


def quote(cents, percent):
    return discounted(cents, percent)
