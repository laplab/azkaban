import time
import sys

import torch.multiprocessing as mp

sys.path.append('../../../')

from azkaban.agent import RandomAgent, A3CAgent, A3CParams, AdvantageMode
from azkaban.env import TeamsEnv, TeamsEnvConf
from azkaban.optim import SharedAdam
from notebooks.a3c_vs_random.parallel_training.model import A3CModel
from notebooks.a3c_vs_random.parallel_training.utils import plot_stats


class A3CParallelRunner(object):
    def __init__(self, conf, params, a3c_team=5, enemy_team=5, n_workers=5, max_ticks=1000):
        self.conf = conf
        self.params = params

        self.a3c_team = a3c_team
        self.enemy_team = enemy_team
        self.n_workers = n_workers
        self.max_ticks = max_ticks

        self.shared_model = self._model_factory()
        self.shared_optimizer = SharedAdam(
            self.shared_model.parameters(),
            lr=self.params.lr
        )
        self.shared_lock = mp.Lock()

    def _model_factory(self):
        return A3CModel(
            in_units=27,
            n_actions=self.conf.action_space.shape()[0]
        )

    def _a3c_agent_factory(self):
        return A3CAgent(
            conf=self.conf,
            params=self.params,
            model=self._model_factory(),
            shared_model=self.shared_model,
            shared_optimizer=self.shared_optimizer,
            trainable=True,
            lock=self.shared_lock
        )

    def _enemy_factory(self):
        return RandomAgent(conf=self.conf)

    def _env_factory(self):
        return TeamsEnv(
            teams=[
                tuple(self._a3c_agent_factory() for _ in range(self.a3c_team)),
                tuple(self._enemy_factory() for _ in range(self.enemy_team))
            ],
            conf=self.conf
        )

    @staticmethod
    def _worker(stats, env, max_ticks):
        while True:
            env.reset()

            for i in range(max_ticks):
                done = env.step(interrupt=(i == max_ticks - 1))

                if done:
                    break

            stats.append(tuple(env.members))

    def run(self):
        manager = mp.Manager()
        stats = manager.list()

        workers = []
        for _ in range(self.n_workers):
            p = mp.Process(target=self._worker, args=(stats, self._env_factory(), self.max_ticks))
            p.start()
            workers.append(p)

        print('Press Ctrl+C to exit')
        try:
            while True:
                time.sleep(1)

                if len(stats) > 0:
                    plot_stats(self.conf.team_names, stats, filename='results.png')
        except KeyboardInterrupt:
            print('Killing workers...')

            for worker in workers:
                worker.terminate()


if __name__ == '__main__':
    conf = TeamsEnvConf(
        world_shape=(7, 7),
        comm_shape=(0,),
        team_names=[
            'a3c',
            'random'
        ]
    )
    params = A3CParams()
    params.advantage_mode = AdvantageMode.GAE

    runner = A3CParallelRunner(conf, params, n_workers=5)
    runner.run()
