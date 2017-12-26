import unittest

import numpy as np

from azkaban.agent import RandomAgent
from azkaban.env.team import TeamsEnv, TeamsEnvConf


class TestStochastic(unittest.TestCase):
    def test_random(self):
        conf = TeamsEnvConf(
            world_shape=(1, 1),
            comm_shape=(1,)
        )
        agent = RandomAgent(conf)
        env = TeamsEnv(
            teams=[[agent]],
            conf=conf
        )

        experiments = 10000
        obs = env._observation((0, 0))

        cells_count, actions_count = env.conf.action_space.shape

        cell_stats = np.zeros(cells_count)
        action_stats = np.zeros(actions_count)

        for _ in range(experiments):
            (cell_id, action), _ = agent.step(obs, 0.0, False)
            cell_stats[cell_id] += 1
            action_stats[action] += 1

        cell_stats /= experiments
        action_stats /= experiments

        cells_ideal = np.ones(cells_count) / cells_count
        actions_ideal = np.ones(actions_count) / actions_count

        self.assertTrue(np.allclose(cells_ideal, cell_stats, atol=0.1))
        self.assertTrue(np.allclose(actions_ideal, action_stats, atol=0.1))


if __name__ == '__main__':
    unittest.main()
