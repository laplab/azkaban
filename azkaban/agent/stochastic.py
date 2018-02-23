import numpy as np

from azkaban.agent.core import Agent


class RandomAgent(Agent):
    def __init__(self, conf):
        self.conf = conf
        self.comm = np.zeros(self.conf.comm_shape)

    def step(self, new_observation, reward, done):
        return self.conf.action_space.sample(), self.comm

    def reset(self):
        pass
