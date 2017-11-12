import uuid


class SparsePlane(object):
    """Sparse plane storing only non-null points"""

    def __init__(self):
        self.points = {}
        self.coords = {}

    def __setitem__(self, coord, point):
        pk = int(uuid.uuid4())

        self.points[pk] = point
        self.coords[coord] = pk

    def __getitem__(self, coord):
        pk = self.coords.get(coord, None)

        if pk is None:
            return None

        return self.points[pk]

    def __delitem__(self, coord):
        pk = self.coords.get(coord, None)

        if pk is None:
            return

        del self.points[pk]
        del self.coords[coord]

    def __contains__(self, coord):
        return coord in self.coords

    def move(self, old, new):
        pk = self.coords.get(old, None)
        del self.coords[old]
        self.coords[new] = pk

    def __iter__(self):
        for coord, pk in self.coords.items():
            yield (coord, self.points[pk])
