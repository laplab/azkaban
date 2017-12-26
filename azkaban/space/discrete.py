from azkaban.space.core import Space
import numpy as np


class Discrete(Space):
    """
    {0, 1, ..., n-1}

    Example usage:
    self.observation_space = spaces.Discrete(2)
    """

    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)

    @property
    def shape(self):
        return (self.n,)

    def contains(self, x):
        return 0 <= x < self.n

    def iter(self):
        yield from range(self.n)
