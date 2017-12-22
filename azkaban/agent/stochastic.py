import random

import numpy as np

from azkaban.agent.core import Agent


class RandomAgent(Agent):
    def __init__(self, conf):
        self.comm = np.zeros(conf.comm_shape)

    def step(self, new_observation, reward, done):
        conf = new_observation.conf
        cell_id = random.choice(range(len(new_observation.view)))
        action = conf.action_space.sample()

        return (cell_id, action), self.comm

    def reset(self):
        pass
