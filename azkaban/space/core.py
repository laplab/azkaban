from abc import abstractmethod


class Space(object):
    @abstractmethod
    def sample(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def contains(self, item):
        pass

    @abstractmethod
    def iter(self):
        pass
