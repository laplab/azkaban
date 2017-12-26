from azkaban.space.core import Space
from itertools import product


class Product(Space):
    def __init__(self, planes):
        if len(planes) != 2:
            raise ValueError('products of only 2 planes are accepted')

        self.planes = planes

    def sample(self):
        result = []
        for plane in self.planes:
            result.append(plane.sample())

        return tuple(result)

    def contains(self, item):
        for idx, coordinate in enumerate(item):
            if not self.planes[idx].contains(coordinate):
                return False

        return True

    def iter(self):
        product_generators = []
        for plane in self.planes:
            product_generators.append(plane.iter())

        yield from product(*product_generators)

    @property
    def shape(self):
        product_shape = []
        for plane in self.planes:
            product_shape += plane.shape

        return tuple(product_shape)
