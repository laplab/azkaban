from abc import abstractmethod


class Space(object):
    @abstractmethod
    def sample(self):
        """Get random action"""
        pass

    @abstractmethod
    def shape(self):
        """Shape of a space"""
        pass

    @abstractmethod
    def elements(self):
        """Get list of elements in space"""
        pass
