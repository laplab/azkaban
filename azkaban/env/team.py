import random
from itertools import product

import numpy as np

from azkaban.display import Table
from azkaban.env.core import Env, Observation, EnvConf
from azkaban.utils import SparsePlane, dataclass, merge_dataclass

TeamsEnvProps = dataclass(
    'TeamsEnvProps',
    health=3,
    actions=0,
    view_radius=1,
    share_radius=1,
    attack_radius=1,
    move_radius=1,
    comm_init=lambda shape: np.random.normal(size=shape)
)


class TeamsEnvConf(merge_dataclass('TeamsEnvConf', (EnvConf, TeamsEnvProps))):
    pass


class TeamsEnv(Env):
    class AgentData(object):
        def __init__(self, agent, team):
            self.agent = agent
            self.team = team

            self.health = None
            self.actions = None
            self.comm = None
            self.reward = None

        def step(self, new_observation, reward, done):
            return self.agent.step(new_observation, reward, done)

        def reset(self, health, actions, comm):
            self.agent.reset()

            self.health = health
            self.actions = actions
            self.comm = comm
            self.reward = 0.0

        @property
        def is_dead(self):
            return self.health == 0

    def __init__(self, teams, conf):
        self.agents = []
        for idx, team in enumerate(teams):
            for agent in team:
                data = self.AgentData(agent, team=idx)
                self.agents.append(data)

        self.conf = conf
        self.init_members = tuple(len(team) for team in teams)
        self.members = None

        self.world = None

        self.img = Table(
            groups_count=len(self.init_members),
            alpha_size=self.conf.health
        )

        self.reset()

    def _offsets(self, radius):
        view_range = range(-radius, radius + 1)
        offsets = list(product(view_range, repeat=len(self.conf.world_shape)))

        return offsets

    def _dist(self, a, b):
        return max(abs(ia - ib) for ia, ib in zip(a, b))

    def _move(self, coord, direction):
        new_coord = []
        for i, dim in enumerate(self.conf.world_shape):
            axis = coord[i] + direction[i]
            new_coord.append(min(dim - 1, max(0, axis)))
        return tuple(new_coord)

    def _view(self, coord):
        view = []
        for offset in self._offsets(self.conf.view_radius):
            new_coord = self._move(coord, offset)

            # return empty communication info if there is
            # no agent or agent stays at the same point
            if coord == new_coord or new_coord not in self.world:
                view.append(np.zeros_like(self.conf.comm_shape))
            else:
                view.append(self.world[new_coord].comm)

        return view

    @property
    def _done(self):
        return sum([x > 0 for x in self.members]) <= 1

    def step(self):
        points = list(self.world)
        random.shuffle(points)

        for coord, agent in points:
            # agents killed on this step
            # are not cleared from points
            if agent.is_dead:
                continue

            agent.actions += 1

            action_radius = max(
                self.conf.share_radius,
                self.conf.attack_radius,
                self.conf.move_radius
            )
            actions = self._offsets(action_radius)
            view = self._view(coord)
            observation = Observation(
                view=view,
                directions=self._offsets(self.conf.view_radius),
                actions=actions,
                conf=self.conf
            )

            action, comm = agent.step(observation, agent.reward, self._done)
            agent.comm = comm
            agent.reward = 0.0

            if action == 0:
                continue

            new_coord = self._move(coord, actions[action - 1])
            if new_coord == coord:
                continue

            dist = self._dist(coord, new_coord)
            neighbour = self.world[new_coord]
            # move
            if neighbour is None and dist <= self.conf.move_radius:
                self.world.move(coord, new_coord)
                agent.actions -= 1
            # share action point
            elif neighbour.team == agent.team and dist <= self.conf.share_radius:
                neighbour.actions += 1
                agent.actions -= 1
            # hit
            elif not neighbour.is_dead and dist <= self.conf.attack_radius:
                neighbour.health -= 1
                agent.actions -= 1

                if neighbour.is_dead:
                    agent.reward = 1.0
                    self.members[neighbour.team] -= 1
                    del self.world[new_coord]

        return self._done

    def reset(self):
        self.img.reset()
        self.world = SparsePlane()
        self.members = list(self.init_members)

        for agent in self.agents:
            coord = None
            while coord is None or coord in self.world:
                coord = tuple(random.randint(0, dim-1) for dim in self.conf.world_shape)

            self.world[coord] = agent
            agent.reset(
                health=self.conf.health,
                actions=self.conf.actions,
                comm=self.conf.comm_init(shape=self.conf.comm_shape)
            )

    def render(self):
        shape = self.conf.world_shape
        if len(shape) != 2:
            raise ValueError('render supports only 2D worlds')

        n, m = shape
        field = np.zeros(shape=shape + (4,))

        for i in range(n):
            for j in range(m):
                if (i, j) in self.world:
                    agent = self.world[i, j]
                    color = self.img.color(group=agent.team, alpha=agent.health)
                else:
                    color = self.img.background

                field[i, j, :] = color

        self.img.update(field)
