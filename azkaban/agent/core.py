from abc import abstractmethod


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
        :return: (action_id, comm): Returns id of action to perform along with communication vector
        """
        pass

    @abstractmethod
    def reset(self):
        """Called by env before the new session starts"""
        pass
