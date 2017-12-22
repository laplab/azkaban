import numpy as np

from azkaban.agent import RandomAgent
from azkaban.agent.core import Agent


class GreedyAgent(Agent):
    def __init__(self, base_action, conf, fallback_agent=None):
        self.base_action = base_action
        self.fallback_agent = fallback_agent or RandomAgent(conf)
        self.comm = np.zeros(conf.comm_shape)

    def _dist(self, a, b):
        return np.linalg.norm(np.subtract(a, b))

    def step(self, new_observation, reward, done):
        coord = new_observation.state.coord
        view = list(enumerate(new_observation.view))
        closest = sorted(view, key=lambda item: self._dist(coord, item[1].coord))

        for idx, visible in closest:
            if self.base_action in visible.actions_state:
                return (idx, self.base_action), self.comm

        return self.fallback_agent.step(new_observation, reward, done)

    def reset(self):
        pass
