import sys

sys.path.append('../../../')

from azkaban.agent import RandomAgent, A3CAgent, A3CParams, AdvantageMode
from azkaban.env import TeamsEnv, TeamsEnvConf
from azkaban.optim import SharedAdam
from experiments.avicenna_vs_random.model import AvicennaModel
from azkaban.utils.parallel import ParallelRunner


class AvicennaParallelRunner(ParallelRunner):
    def __init__(self, conf, params, *args, **kwargs):
        self.conf = conf
        self.params = params

        super(AvicennaParallelRunner, self).__init__(*args, **kwargs)

    def create_optimizer(self):
        return SharedAdam(
            self.shared_model.parameters(),
            lr=self.params.lr
        )

    def create_model(self):
        return AvicennaModel(
            obs_size=18,
            comm_size=self.conf.comm_shape[0],
            n_actions=self.conf.action_space.shape()[0],
        )

    def create_buddy(self):
        return A3CAgent(
            conf=self.conf,
            params=self.params,
            model=self.create_model(),
            shared_model=self.shared_model,
            shared_optimizer=self.shared_optimizer,
            trainable=True,
            lock=self.shared_lock
        )

    def create_enemy(self):
        return RandomAgent(conf=self.conf)

    def create_env(self):
        return TeamsEnv(
            teams=[
                tuple(self.create_buddy() for _ in range(self.n_buddies)),
                tuple(self.create_enemy() for _ in range(self.n_enemies))
            ],
            conf=self.conf
        )


if __name__ == '__main__':
    conf = TeamsEnvConf(
        world_shape=(10, 10),
        comm_shape=(16,),
    )
    params = A3CParams()
    params.advantage_mode = AdvantageMode.GAE

    runner = AvicennaParallelRunner(conf, params, n_buddies=7, n_workers=1)
    runner.run()
