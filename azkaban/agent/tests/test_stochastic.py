import unittest

import numpy as np

from azkaban.core import Observation
from azkaban.env.team import TeamsEnvConf
from azkaban.agent import RandomAgent


class TestStochastic(unittest.TestCase):
    def test_random(self):
        actions_count = 10
        experiments = 10000

        obs = Observation(
            view=[(0,)] * actions_count,
            directions=[(0, 0)] * actions_count,
            actions=[0] * (actions_count - 1),
            conf=TeamsEnvConf(
                world_shape=(1, 1),
                comm_shape=(0,)
            )
        )
        agent = RandomAgent()
        stats = np.zeros(actions_count)
        for _ in range(experiments):
            stats[agent.step(obs, 0.0, False)[0]] += 1

        stats /= experiments
        ideal = np.ones(actions_count) / actions_count

        self.assertTrue(np.allclose(stats, ideal, atol=0.01))


if __name__ == '__main__':
    unittest.main()
