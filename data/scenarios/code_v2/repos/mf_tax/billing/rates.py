def tax_cents(price_cents, rate_bps):
    return price_cents * rate_bps // 10000
