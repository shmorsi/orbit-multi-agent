from billing.rates import tax_cents


def charge(price_cents, rate_bps):
    taxed = price_cents + tax_cents(price_cents, rate_bps)
    return taxed + tax_cents(taxed, rate_bps)
