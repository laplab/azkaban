import random

import numpy as np

from azkaban.agent.core import Agent


class RandomAgent(Agent):
    def __init__(self, conf):
        self.comm = np.zeros(conf.comm_shape)

    def step(self, new_observation, reward, done):
        return new_observation.conf.action_space.sample(), self.comm

    def reset(self):
        pass
