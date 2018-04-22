import os
from abc import abstractmethod

import torch.multiprocessing as mp

from azkaban.utils.helpers import TBSummaryWriter


class ParallelTBSummaryWriter(TBSummaryWriter):
    def __init__(self, *args, **kwargs):
        super(ParallelTBSummaryWriter, self).__init__(*args, **kwargs)
        self.step = 0

    def add_members(self, stats):
        self.step += 1
        buddy, enemy = stats
        self.log_scalar('alive/buddy', buddy, self.step)
        self.log_scalar('alive/enemy', enemy, self.step)


class ParallelRunner(object):
    def __init__(self, n_buddies=5, n_enemies=5, n_workers=5, max_ticks=1000,
                 base_dir='logs', buddy_name='buddy', enemy_name='enemy'):
        """
        Runner of parallel experiments
        :param n_buddies: Number of team members for training
        :param n_enemies: Number of enemies
        :param n_workers: Number of workers to execute environments in
        :param max_ticks: Maximum count of ticks environment can exist
        :param filename: Filename to render charts in
        """
        self.n_buddies = n_buddies
        self.n_enemies = n_enemies
        self.n_workers = n_workers

        self.buddy_name = buddy_name
        self.enemy_name = enemy_name

        self.max_ticks = max_ticks

        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)

        runs = os.listdir(base_dir)
        max_index = 0
        for run in runs:
            full_path = os.path.join(base_dir, run)
            if os.path.isdir(full_path) and run.startswith('run_'):
                parts = run.split('_')
                if len(parts) > 1:
                    try:
                        index = int(parts[-1])
                        max_index = max(max_index, index)
                    except ValueError:
                        continue

        log_dir = os.path.join(base_dir, 'run_{}'.format(max_index + 1))
        self.writer = ParallelTBSummaryWriter(log_dir)

        self.shared_lock = mp.Lock()
        self.shared_model = self.create_model()
        self.shared_model.share_memory()
        self.shared_optimizer = self.create_optimizer()
        self.shared_optimizer.share_memory()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_buddy(self):
        pass

    @abstractmethod
    def create_enemy(self):
        pass

    @abstractmethod
    def create_env(self):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @staticmethod
    def _worker(queue, env, max_ticks):
        while True:
            env.reset()

            for i in range(max_ticks):
                done = env.step(interrupt=(i == max_ticks - 1))

                if done:
                    break

            queue.put(tuple(env.members))

    def run(self):
        queue = mp.Queue()

        workers = []
        for _ in range(self.n_workers):
            p = mp.Process(
                target=self._worker,
                args=(
                    queue,
                    self.create_env(),
                    self.max_ticks
                )
            )
            p.start()
            workers.append(p)

        print('Press Ctrl+C to exit')
        try:
            while True:
                members = queue.get()
                self.writer.add_members(members)
        except KeyboardInterrupt:
            print('Killing workers...')

            for worker in workers:
                worker.terminate()

            self.writer.close()
