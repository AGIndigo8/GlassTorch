import itertools

class TuplingIterator:
    """Iterates over all tuples of length tuple_size with values in range(N).

    If no_repeats=True (default), skips tuples that contain duplicate values.
    """

    def __init__(self, tuple_size: int, N: int, no_repeats: bool = True):
        self.tuple_size = tuple_size
        self.N = N
        self.no_repeats = no_repeats

    def __iter__(self):
        if self.no_repeats:
            return itertools.combinations(range(self.N), self.tuple_size)
        else:
            return itertools.product(range(self.N), repeat=self.tuple_size)