from azkaban.space.core import Space
import random


class Discrete(Space):
    """
    Represents space of ids of items in provided iterable
    """

    def __init__(self, iterable):
        self.items = iterable

        self.size = len(iterable)
        self.ids = tuple(range(self.size))

    def sample(self):
        return random.choice(self.ids)

    def shape(self):
        return (self.size,)

    def elements(self):
        return self.ids

    def get(self, id):
        """
        Get item by it's id

        NOTE: Normally SHOULDN'T be called by an agent
        """
        return self.items[id]
