def parse_mdy(text):
    month, day, year = text.split('/')
    return (int(year), int(day), int(month))
