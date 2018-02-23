from abc import abstractmethod

from azkaban.error import ArgumentError


class Env(object):
    """Interface for an environment"""

    @abstractmethod
    def step(self, interrupt):
        """
        Make one time tick in the world

        :return: True if game has ended and False otherwise
        """
        pass

    @abstractmethod
    def reset(self):
        """Prepares env for the new session. Calls `agent.reset` for every agent in this env"""
        pass

    @abstractmethod
    def render(self):
        """Displays current state of the world"""
        pass


class EnvConf(object):
    """
    Base class for all configs

    Required attributes to specify are:
    :var world_shape: Shows shape of the world
    :var comm_shape Shows shape of communication vector

    Required to be provided after __init__ call attributes are:
    :var observation_shapes: Tuple of tuples each representing shape of observation unit
                             For example suppose agent receives observation of 2 matrices 2x2
                             and one of 2x4x10. Then observation_shapes will be equal to
                             ((2, 2), (2, 2), (2, 4, 10))
    :var action_space: Finite iterable containing possible actions
    """

    def __init__(self, *args, **kwargs):
        self.world_shape = None
        self.comm_shape = None
        self.observation_shapes = None
        self.action_space = None

        if len(args) > 0:
            raise ArgumentError(
                'Positional arguments are not allowed for {}'.format(self.__class__.__name__)
            )

        if 'world_shape' not in kwargs or 'comm_shape' not in kwargs:
            raise ArgumentError('All EnvConfs require `world_shape` and `comm_shape` in kwargs')

        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ArgumentError(
                    'Unexpected kwarg {} for {}'.format(key, self.__class__.__name__)
                )
