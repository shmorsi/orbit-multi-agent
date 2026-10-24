def discounted(cents, percent):
    once = cents - (cents * percent // 100)
    return once - (once * percent // 100)
