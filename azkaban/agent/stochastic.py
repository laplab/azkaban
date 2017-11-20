import random

import numpy as np

from azkaban.agent.core import Agent


class RandomAgent(Agent):
    def step(self, new_observation, reward, done):
        actions = new_observation.actions
        conf = new_observation.conf
        return random.choice(range(len(actions) + 1)), np.random.random(conf.comm_shape)

    def reset(self):
        pass
