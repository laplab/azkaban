from abc import abstractmethod


class Canvas(object):
    @abstractmethod
    def update(self, values):
        pass

    @abstractmethod
    def reset(self):
        pass
