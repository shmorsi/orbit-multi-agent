from billing.rates import tax_cents


def charge(price_cents, rate_bps):
    return price_cents + tax_cents(price_cents, rate_bps)
