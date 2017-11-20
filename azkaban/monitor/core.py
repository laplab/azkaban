from abc import abstractmethod

from azkaban.env.core import Env


class Monitor(Env):
    @abstractmethod
    def finish(self):
        pass
