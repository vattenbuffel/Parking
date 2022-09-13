from collections import deque

class RollingAverage:
    def __init__(self, n) -> None:
        self.n = n
        self.queue = deque(maxlen=n)

    def update(self, value):
        self.queue.append(value)

    def get(self):
        return sum(self.queue) / self.n