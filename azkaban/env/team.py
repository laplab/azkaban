import random
from enum import Enum
from itertools import product

import numpy as np
from scipy.spatial.distance import euclidean

from azkaban.display import Table
from azkaban.env.core import Env, EnvConf
from azkaban.error import ArgumentError
from azkaban.space import Discrete
from azkaban.agent.a3c import A3CAgent


class TeamsMap(object):
    def __init__(self, conf):
        self.conf = conf

        self.used = np.zeros(self.conf.world_shape, dtype=np.bool)
        self.team = np.zeros(self.conf.world_shape, dtype=np.int)
        self.health = np.zeros(self.conf.world_shape, dtype=np.int)
        self.actions = np.zeros(self.conf.world_shape, dtype=np.int)
        self.cur_reward = np.zeros(self.conf.world_shape, dtype=np.float)
        self.prev_reward = np.zeros(self.conf.world_shape, dtype=np.float)
        self.comm = np.zeros(self.conf.world_shape + self.conf.comm_shape,
                             dtype=np.float)
        self.dcomm = np.zeros(self.conf.world_shape + self.conf.comm_shape,
                              dtype=np.float)

        self.reset()

    def move(self, source, dest):
        self.used[dest] = self.used[source]
        self.team[dest] = self.team[source]
        self.health[dest] = self.health[source]
        self.actions[dest] = self.actions[source]
        self.cur_reward[dest] = self.cur_reward[source]
        self.prev_reward[dest] = self.prev_reward[source]
        self.comm[dest] = self.comm[source]
        self.dcomm[dest] = self.dcomm[source]

        self.clear(source)

    def clear(self, coord):
        self.used[coord] = False
        self.team[coord] = 0
        self.health[coord] = 0
        self.actions[coord] = 0
        self.cur_reward[coord] = 0
        self.prev_reward[coord] = 0
        self.comm[coord].fill(0)
        self.dcomm[coord].fill(0)

    def reset(self):
        self.used.fill(False)

        self.team.fill(0)
        self.health.fill(0)
        self.actions.fill(0)
        self.cur_reward.fill(0)
        self.prev_reward.fill(0)
        self.comm.fill(0)
        self.dcomm.fill(0)


class TeamsActions(Enum):
    NO_OP = 0
    SHARE = 1
    ATTACK = 2
    MOVE = 3


class TeamsEnvConf(EnvConf):
    def __init__(self, *args, **kwargs):
        # initial state of agents
        self.health = 3
        self.actions = 0

        # agents properties
        self.view_radius = 1
        self.share_radius = 1
        self.attack_radius = 1
        self.move_radius = 1

        # maximum distance agent can reach by making action
        self.act_radius = None

        # optional list of names for each team
        self.team_names = None

        # reward description
        self.kill_reward = 1.0
        self.damage_reward = 0.1
        self.lost_health_reward = -0.5
        self.time_tick_reward = -0.01

        super(TeamsEnvConf, self).__init__(*args, **kwargs)

        self.act_radius = max(self.share_radius, self.attack_radius, self.move_radius)

        if self.view_radius < self.act_radius:
            raise ArgumentError('Agent cannot act blindly, extend view radius')

        width = 2 * self.view_radius + 1
        self.observation_shapes = (
            (width, width),                   # teams
            (width, width),                   # health indicators
            (width, width),                   # actions left
            (width, width) + self.comm_shape  # communication vectors
        )

        depth = len(self.world_shape)
        directions = tuple(product(range(-self.act_radius, self.act_radius + 1), repeat=depth))
        actions = tuple(range(len(TeamsActions)))

        self.action_space = Discrete(tuple(product(directions, actions)))


class TeamsEnv(Env):
    def __init__(self, teams, conf):
        self.agents = []
        self.teams_count = len(teams)
        for idx, team in enumerate(teams):
            for agent in team:
                self.agents.append({
                    'team': idx,
                    'actor': agent,
                    'coord': None
                })

        self.conf = conf
        self.map = TeamsMap(self.conf)

        self.members = None
        self.interrupted = None

        self.img = Table(
            groups_count=self.teams_count,
            alpha_size=self.conf.health,
            group_labels=self.conf.team_names
        )

        self.stats = []

        self.reset()

    def _visible_buddies(self, center):
        radius = self.conf.view_radius

        team = self.map.team[center]
        buddies = []
        for agent in self.agents:
            coord = agent['coord']
            if self.map.team[coord] == team and coord != center:
                buddies.append(coord)

        return buddies

    def _observation(self, center):
        radius = self.conf.view_radius
        indices = []
        for ax in center:
            indices.append(slice(ax-radius, ax+radius + 1))

        return (self.map.team[indices],
                self.map.health[indices],
                self.map.actions[indices])

    @property
    def _done(self):
        return self.interrupted or sum([x > 0 for x in self.members]) <= 1

    def _move(self, coord, direction):
        result = list(coord)
        for (i, delta), ax in zip(enumerate(direction), self.conf.world_shape):
            result[i] += delta

            if not (self.conf.view_radius <= result[i] <= ax - self.conf.view_radius - 1):
                return None, False

        return tuple(result), True

    def step(self, interrupt=False):
        self.interrupted = interrupt

        random.shuffle(self.agents)

        if len(self.stats) > 0 and len(self.stats) % 500 == 0:
            stats = np.array(self.stats)
            print_array = lambda x: print(np.array2string(x, threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', ''))

            mean = np.mean(stats, axis=0)
            std = np.std(stats, axis=0)

            print('-' * 100)
            print('{} communication vectors collected'.format(len(self.stats)))

            print('Mean norm:', np.linalg.norm(mean))
            print('Mean vector:')
            print_array(mean)

            print('Std norm:', np.linalg.norm(std))
            print('Std vector:')
            print_array(std)

            print('-' * 100)
            print()

        for agent in self.agents:
            coord, actor = agent['coord'], agent['actor']

            # check if agent is dead on this step
            if not self.map.used[coord]:
                continue

            self.map.actions[coord] += 1
            obs = self._observation(coord)

            buddies = self._visible_buddies(coord)
            comms = [self.map.comm[bcoord] for bcoord in buddies]

            # TODO: FIX if done happened on the previous step other agents are not notified
            action_id, comm, dcomms = actor.step(
                obs,
                self.map.prev_reward[coord],
                comms,
                self.map.dcomm[coord],
                self._done
            )

            if isinstance(actor, A3CAgent):
                self.stats.append(comm)

            self.map.comm[coord] = comm
            for bcoord, dcomm in zip(buddies, dcomms):
                self.map.dcomm[bcoord] += dcomm

            direction, action = self.conf.action_space.get(action_id)
            new_coord, success = self._move(coord, direction)

            # new coord is out of bounds
            if not success:
                continue

            # don't want to do anything
            if action == TeamsActions.NO_OP.value:
                continue

            # Move
            # check if cell is free
            if action == TeamsActions.MOVE.value and not self.map.used[new_coord]:
                self.map.move(coord, new_coord)
                agent['coord'] = new_coord

            # Share
            # check if there is anybody to share with from the same team
            elif (action == TeamsActions.SHARE.value and
                  self.map.used[new_coord] and
                  self.map.team[coord] == self.map.team[new_coord]):
                self.map.actions[new_coord] += 1

            # Attack
            # check if there is anybody to share with from different team
            elif (action == TeamsActions.ATTACK.value and
                  self.map.used[new_coord] and
                  self.map.team[coord] != self.map.team[new_coord]):
                self.map.health[new_coord] -= 1
                self.map.cur_reward[new_coord] += self.conf.lost_health_reward

                if self.map.health[new_coord] == 0:
                    # TODO: make faster for large number of agents
                    for neighbour in self.agents:
                        if neighbour['coord'] == new_coord:
                            new_obs = self._observation(new_coord)

                            buddies = self._visible_buddies(new_coord)
                            comms = [self.map.comm[bcoord] for bcoord in buddies]
                            neighbour['actor'].step(
                                new_obs,
                                self.map.prev_reward[new_coord],
                                comms,
                                self.map.dcomm[new_coord],
                                True
                            )
                            break

                    self.members[self.map.team[new_coord]] -= 1
                    self.map.clear(new_coord)

                    self.map.cur_reward[coord] += self.conf.kill_reward
                else:
                    self.map.cur_reward[coord] += self.conf.damage_reward

            # invalid action taken
            else:
                pass

            self.map.prev_reward[coord] = self.map.cur_reward[coord]
            self.map.cur_reward[coord] = 0

            self.map.actions[coord] -= 1

        # notify all alive agents in case of win
        if self._done:
            # TODO: make faster for large number of agents
            for agent in self.agents:
                coord, actor = agent['coord'], agent['actor']

                if self.map.health[coord] == 0:
                    continue

                obs = self._observation(coord)

                buddies = self._visible_buddies(coord)
                comms = [self.map.comm[bcoord] for bcoord in buddies]

                actor.step(
                    obs,
                    self.map.prev_reward[coord],
                    comms,
                    self.map.dcomm[coord],
                    True
                )

        return self._done

    def reset(self):
        self.img.reset()
        self.map.reset()
        self.members = [0] * self.teams_count
        self.interrupted = False

        radius = self.conf.view_radius
        for agent in self.agents:
            coord = None
            while coord is None or self.map.used[coord]:
                coord = tuple(
                    random.randint(radius, dim - 1 - radius) for dim in self.conf.world_shape
                )

            agent['actor'].reset()
            agent['coord'] = coord
            self.members[agent['team']] += 1

            self.map.used[coord] = True
            self.map.team[coord] = agent['team']
            self.map.health[coord] = self.conf.health
            self.map.actions[coord] = self.conf.actions

    def render(self):
        shape = self.conf.world_shape
        if len(shape) != 2:
            raise ValueError('render supports only 2D worlds')

        n, m = shape
        field = np.zeros(shape=shape + (4,))

        for i in range(n):
            for j in range(m):
                if self.map.used[(i, j)]:
                    color = self.img.color(
                        group=self.map.team[(i, j)],
                        alpha=self.map.health[(i, j)]
                    )
                else:
                    color = self.img.background

                field[i, j, :] = color

        self.img.update(field)
