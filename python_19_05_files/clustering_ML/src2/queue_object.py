import heapq
import numpy as np

class PriorityQueue:
    def __init__(self):
        # Initialize an empty list to serve as the heap
        self._queue = []

    def push_func(self, score, node):
        """Pushes a new (score, node) tuple onto the heap."""
        heapq.heappush(self._queue, (score, node))

    def push_mult(self, pairs):
        """Pushes an array of (score, node) pairs onto the heap."""
        for score, node in pairs:
            heapq.heappush(self._queue, (score, node))

    def extract_lowest_score(self):
        """Pops and returns the (score, node) tuple with the lowest score."""
        # Note: Your algorithm checks if it's empty before calling this, 
        # so we don't need to handle IndexError here unless you want to be extra safe.
        return heapq.heappop(self._queue)

    def is_empty(self):
        """Returns True if the queue is empty, False otherwise."""
        return len(self._queue) == 0

    def peek_lowest_score(self):
        """Returns the (score, node) tuple with the lowest score without removing it."""
        return self._queue[0]