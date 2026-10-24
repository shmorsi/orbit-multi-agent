class Counter:
    def __init__(self):
        self.counts = {}

    def add(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def get(self, key):
        return self.counts.get(key, 0)
