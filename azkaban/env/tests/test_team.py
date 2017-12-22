import unittest

from azkaban.agent import GreedyAgent
from azkaban.env.team import TeamsEnv, TeamsEnvConf, TeamsActions


class TestTeam(unittest.TestCase):
    def test_action_sharing(self):
        conf = TeamsEnvConf(
            world_shape=(2, 1),
            comm_shape=(1,)
        )
        env = TeamsEnv(
            teams=[[
                GreedyAgent(conf=conf, base_action=TeamsActions.SHARE),
                GreedyAgent(conf=conf, base_action=TeamsActions.NO_OP)
            ]],
            conf=conf
        )
        env.step()

        self.assertEqual(env.agents[0][0].data.actions, 0)
        self.assertEqual(env.agents[0][1].data.actions, 2)

    def test_damage(self):
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

    def test_reward(self):
        conf = TeamsEnvConf(
            world_shape=(3, 1),
            comm_shape=(1,),
            health=1,
            kill_reward=1.0,
            damage_reward=0,
            lost_health_reward=0,
            time_tick_reward=0
        )
        env = TeamsEnv(
            teams=[
                [GreedyAgent(conf=conf, base_action=TeamsActions.ATTACK)],
                [
                    GreedyAgent(conf=conf, base_action=TeamsActions.NO_OP),
                    GreedyAgent(conf=conf, base_action=TeamsActions.NO_OP)
                ]
            ],
            conf=conf
        )
        done = env.step()

        self.assertFalse(done)
        self.assertEqual(env.agents[0][0].prev_reward, 1.0)
        self.assertEqual(env.agents[1][0].prev_reward, 0.0)
        self.assertEqual(env.agents[1][1].prev_reward, 0.0)

    def test_win(self):
        conf = TeamsEnvConf(
            world_shape=(2, 1),
            comm_shape=(1,),
            health=1
        )
        env = TeamsEnv(
            teams=[
                [GreedyAgent(conf=conf, base_action=TeamsActions.ATTACK)],
                [GreedyAgent(conf=conf, base_action=TeamsActions.NO_OP)]
            ],
            conf=conf
        )
        done = env.step()

        self.assertTrue(done)
        self.assertEqual(env.members, [1, 0])


if __name__ == '__main__':
    unittest.main()
