from abc import abstractmethod

from azkaban.env.core import Env


class Monitor(Env):
    """Base class for monitoring an environment"""

    @abstractmethod
    def finish(self):
        """
        Calling this method signals that `Monitor` must finish
        all his tasks (close file descriptors, connections, etc.)
        """
        pass
