from abc import abstractmethod

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


class EnvConf(dataclass('EnvConf', world_shape=None, comm_shape=None, observation_space=None, action_space=None)):
    """
    Base class for all configs

    :var world_shape: Shows shape of the world
    :var comm_shape Shows shape of communication vector
    :var action_space: `Space` of observations available in the world
    :var action_space: `Space` of actions available in the world
    """
    pass
