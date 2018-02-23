import unittest

import numpy as np

from azkaban.agent import RandomAgent
from azkaban.env.team import TeamsEnv, TeamsEnvConf


class TestStochastic(unittest.TestCase):
    def test_random(self):
        conf = TeamsEnvConf(
            world_shape=(5, 5),
            comm_shape=(1,)
        )
        agent = RandomAgent(conf)
        env = TeamsEnv(
            teams=[[agent]],
            conf=conf
        )

        actions_count, = conf.action_space.shape()
        experiments = 10000
        obs = env._observation((0, 0))

        stats = np.zeros(actions_count)
        for _ in range(experiments):
            action_id, _ = agent.step(obs, 0.0, False)
            stats[action_id] += 1

        stats /= experiments
        stats_ideal = np.ones(actions_count) / actions_count

        self.assertTrue(np.allclose(stats_ideal, stats, atol=0.1))


if __name__ == '__main__':
    unittest.main()
