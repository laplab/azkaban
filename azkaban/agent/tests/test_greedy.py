import unittest

from azkaban.env import TeamsEnv, TeamsEnvConf, TeamsActions
from azkaban.agent import GreedyAgent


class TestGreedy(unittest.TestCase):
    def test_greed(self):
        conf = TeamsEnvConf(
            world_shape=(2, 1),
            comm_shape=(1,),
            health=3
        )
        env = TeamsEnv(
            teams=[
                [GreedyAgent(conf=conf, base_action=TeamsActions.ATTACK)],
                [GreedyAgent(conf=conf, base_action=TeamsActions.NO_OP)]
            ],
            conf=conf
        )
        done = env.step()

        self.assertFalse(done)
        self.assertEqual(env.agents[0][0].data.health, 3)
        self.assertEqual(env.agents[1][0].data.health, 2)
