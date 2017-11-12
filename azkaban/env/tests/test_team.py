import unittest

import numpy as np

from azkaban.core import Agent
from azkaban.env.team import TeamsEnv, TeamsEnvConf


class TestAgent(Agent):
    def __init__(self, neutral=False):
        self.neutral = neutral

    def step(self, obs, reward, done):
        action = 0
        if not self.neutral:
            for comm, dir in zip(obs.view, obs.directions):
                if np.all(comm > 0) and dir in obs.actions:
                    action = obs.actions.index(dir)
        return action, np.ones(1)

    def reset(self):
        pass


class TestTeam(unittest.TestCase):
    def test_action_sharing(self):
        env = TeamsEnv(
            teams=[[TestAgent(), TestAgent(neutral=True)]],
            conf=TeamsEnvConf(
                world_shape=(2, 1),
                comm_shape=(1,),
                comm_init=lambda shape: np.ones(shape)
            )
        )
        env.step()

        self.assertEqual(env.agents[0].actions, 0)
        self.assertEqual(env.agents[1].actions, 2)

    def test_damage(self):
        env = TeamsEnv(
            teams=[[TestAgent()], [TestAgent(neutral=True)]],
            conf=TeamsEnvConf(
                world_shape=(2, 1),
                comm_shape=(1,),
                health=3,
                comm_init=lambda shape: np.ones(shape)
            )
        )
        done = env.step()

        self.assertFalse(done)
        self.assertEqual(env.agents[0].health, 3)
        self.assertEqual(env.agents[1].health, 2)

    def test_reward(self):
        env = TeamsEnv(
            teams=[[TestAgent()], [TestAgent(neutral=True), TestAgent(neutral=True)]],
            conf=TeamsEnvConf(
                world_shape=(3, 1),
                comm_shape=(1,),
                health=1,
                comm_init=lambda shape: np.ones(shape)
            )
        )
        done = env.step()

        self.assertFalse(done)
        self.assertEqual(env.agents[0].reward, 1.0)
        self.assertEqual(env.agents[1].reward, 0.0)
        self.assertEqual(env.agents[2].reward, 0.0)

    def test_win(self):
        env = TeamsEnv(
            teams=[[TestAgent()], [TestAgent(neutral=True)]],
            conf=TeamsEnvConf(
                world_shape=(2, 1),
                comm_shape=(1,),
                health=1,
                comm_init=lambda shape: np.ones(shape)
            )
        )
        done = env.step()

        self.assertTrue(done)
        self.assertEqual(env.members, [1, 0])


if __name__ == '__main__':
    unittest.main()
