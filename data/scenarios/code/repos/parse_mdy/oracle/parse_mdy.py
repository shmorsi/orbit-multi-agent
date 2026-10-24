def parse_mdy(text):
    month, day, year = text.split('/')
    return (int(year), int(month), int(day))
