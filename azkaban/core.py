from abc import abstractmethod

from azkaban.utils import dataclass


class Agent(object):
    """Interface for an agent"""

    @abstractmethod
    def step(self, new_observation, reward, done):
        """
        Called by env to get an action from the agent

        :param new_observation: Observation after the last action (initial world state at the session start)
        :param reward: Reward after the last action (0.0 at the session start)
        :param done: True only if game ended or agent is killed (if False, this
                     is the last call of step in this session)
        :return: (action, message): Returns action id and a new message vector
        """
        pass

    @abstractmethod
    def reset(self):
        """Called by env before the new session starts"""
        pass


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
          3.3 perform an action and update message vector
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


class Observation(dataclass('Observation', view=None, directions=None, actions=None, conf=None)):
    """
    Base class for observations that agents receive

    :var view: communication vector of surrounding cells
    :var directions: for each communication vector in `view` defines
                     direction in which its cell is located
    :var actions: list of available actions. agent returns index (starting from 1)
                  of item from this list to perform an action or zero to do nothing
    :var conf: configuration of current env
    """
    pass


class EnvConf(dataclass('EnvConf', world_shape=None, comm_shape=None)):
    """Base class for all configs"""
    pass
