import random
from abc import abstractmethod, abstractclassmethod

from azkaban.utils import dataclass


class Env(object):
    """Interface for an environment"""

    @abstractmethod
    def step(self):
        """
        Make one time tick in the world. Env does the following:

        1. Random shuffle agent list
        2. Give action points according to the policy configured
        3. For each agent:
          3.1 if agent is dead, continue
          3.2 call agent.step
          3.3 perform an action and update communication vector
        4. Return True if game has ended and False otherwise

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


class Observation(dataclass('Observation', view=None, state=None, conf=None)):
    """
    Base class for observations that agents receive

    :var view: List of `ObservationCell` objects representing surrounding cells
    :var state: `ObservationCell` of the cell agent is located on
    :var conf: Configuration of current env
    """
    pass


class ObservationCell(dataclass('ObservationCell', coord=None, actions_state=None)):
    """
    Item of `Observation`s `view`

    :var coord: Cell coordinates in world
    :var actions_state: Mapping between actions and bool values representing if they
                        are available for this cell
    """
    pass


class EnvConf(dataclass('EnvConf', world_shape=None, comm_shape=None, action_space=None)):
    """
    Base class for all configs

    :var world_shape: Shows shape of the world
    :var comm_shape Shows shape of communication vector
    :var action_space: `ActionSpace` of the world
    """
    pass


class ActionSpace(object):
    """Base class for all action spaces"""

    @abstractclassmethod
    def sample(cls):
        pass

    @abstractclassmethod
    def __contains__(cls, item):
        pass


class Discrete(ActionSpace):
    """
    Discrete action spaces are defined as:
    1) `len(discrete_space)` is a finite positive integer representing
       a number of actions in this space
    2) each number in range `0..len(discrete_space)-1` **must** represent a valid action
    3) action #0 is a "do nothing" action

    Advice: Discrete action spaces could be described as `Enum` objects (see `TeamsActions` for an example)
    """

    @abstractclassmethod
    def __len__(cls):
        pass

    @abstractclassmethod
    def __iter__(cls):
        pass

    @classmethod
    def sample(cls):
        return random.choice(tuple(cls))
