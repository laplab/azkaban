import random
from enum import Enum
from functools import lru_cache
from itertools import product

import numpy as np

from azkaban.display import Table
from azkaban.env.core import Env, Observation, ObservationCell, EnvConf, Discrete
from azkaban.utils import SparsePlane, dataclass, merge_dataclass, HiddenFrozenView


class TeamsAccess(Enum):
    BLOCKED = 'no_access'
    SAME_TEAM = 'same_team'
    ALL_TEAMS = 'all_teams'


class TeamsActions(Discrete, Enum):
    NO_OP = 0
    SHARE = 1
    ATTACK = 2
    MOVE = 3


TeamsEnvProps = dataclass(
    'TeamsEnvProps',
    health=3,
    actions=0,
    view_radius=1,
    share_radius=1,
    attack_radius=1,
    move_radius=1,
    see_health=TeamsAccess.SAME_TEAM,
    see_actions=TeamsAccess.SAME_TEAM,
    see_comm=TeamsAccess.SAME_TEAM,
    comm_init=lambda shape: np.random.normal(size=shape),
    team_names=None,
    kill_reward=10.0,
    damage_reward=1.0,
    lost_health_reward=-2.0,
    time_tick_reward=-0.1
)


class TeamsEnvConf(merge_dataclass('TeamsEnvConf', (EnvConf, TeamsEnvProps), action_space=TeamsActions)):
    pass


TeamsObservableProps = dataclass(
    'TeamsObservableProps',
    team=None,
    health=None,
    actions=None,
    comm=None,
)

TeamsObservationCell = merge_dataclass('TeamsObservationCell', (ObservationCell, TeamsObservableProps))


class TeamsDataCell(dataclass('TeamsDataCell', agent=None, data=None, prev_reward=0.0, cur_reward=0.0)):
    def step(self, new_observation, reward, done):
        return self.agent.step(new_observation, reward, done)

    def reset(self, *args, **kwargs):
        self.agent.reset()
        self.prev_reward = 0.0
        self.cur_reward = 0.0

        self.data = self.data.__class__(*args, **kwargs)

    @property
    def is_dead(self):
        return self.data.health == 0


class TeamsEnv(Env):
    def __init__(self, teams, conf):
        self.agents = {}
        for idx, team in enumerate(teams):
            self.agents[idx] = []

            for agent in team:
                data = TeamsDataCell(agent, data=TeamsObservationCell())
                self.agents[idx].append(data)

        self.conf = conf
        self.members = None
        self.world = None

        self.img = Table(
            groups_count=len(self.agents),
            alpha_size=self.conf.health,
            group_labels=self.conf.team_names
        )

        self.reset()

    def _coords_around(self, coord):
        depth = len(self.conf.world_shape)
        radius = self.conf.view_radius
        segment = range(-radius, radius + 1)

        for offset in product(segment, repeat=depth):
            dist = max(map(abs, offset))
            if dist == 0:
                continue

            is_valid = True
            new_coord = []
            for i, dim in enumerate(self.conf.world_shape):
                axis = coord[i] + offset[i]
                is_valid &= 0 <= axis < dim
                new_coord.append(axis)

            yield tuple(new_coord), dist, is_valid

    @lru_cache(10)
    def _actions_state(self, can_move=False, can_share=False, can_attack=False):
        state = {TeamsActions.NO_OP}

        if can_move:
            state.add(TeamsActions.MOVE)
        
        if can_share:
            state.add(TeamsActions.SHARE)

        if can_attack:
            state.add(TeamsActions.ATTACK)

        return frozenset(state)

    def _observation(self, center):
        state = self.world[center].data
        state.coord = center

        view = []
        for remote, dist, is_valid in self._coords_around(center):
            neighbour = self.world[remote]

            if neighbour is None or not is_valid:
                visible = TeamsObservationCell(
                    coord=remote,
                    actions_state=self._actions_state(
                        can_move=(is_valid and dist <= self.conf.move_radius)
                    )
                )
            else:
                visible = neighbour.data
                visible.coord = remote
                visible.actions_state = self._actions_state(
                    can_share=(state.team == neighbour.data.team and dist <= self.conf.share_radius),
                    can_attack=(state.team != neighbour.data.team and dist <= self.conf.attack_radius)
                )

            hidden_fields = []
            if ((self.conf.see_health == TeamsAccess.SAME_TEAM and state.team != visible.team) or
                 self.conf.see_health == TeamsAccess.BLOCKED):
                hidden_fields.append('health')

            if ((self.conf.see_actions == TeamsAccess.SAME_TEAM and state.team != visible.team) or
                 self.conf.see_actions == TeamsAccess.BLOCKED):
                hidden_fields.append('actions')

            if ((self.conf.see_comm == TeamsAccess.SAME_TEAM and state.team != visible.team) or
                 self.conf.see_comm == TeamsAccess.BLOCKED):
                hidden_fields.append('comm')

            visible = HiddenFrozenView(visible, mask=set(hidden_fields))
            view.append(visible)

        observation = Observation(
            view=view,
            state=state,
            conf=self.conf
        )

        return observation

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

            agent.data.actions += 1
            obs = self._observation(coord)
            (cell_id, action), comm = agent.step(obs, agent.prev_reward, self._done)
            visible = obs.view[cell_id]
            neighbour = self.world[visible.coord]

            agent.comm = comm
            agent.cur_reward += self.conf.time_tick_reward

            if action == TeamsActions.NO_OP:
                continue

            # action is not available
            if action not in visible.actions_state:
                continue

            # move
            if action == TeamsActions.MOVE:
                self.world.move(coord, visible.coord)
            # share action point
            elif action == TeamsActions.SHARE:
                neighbour.data.actions += 1
            # but most importantly he attac
            elif action == TeamsActions.ATTACK:
                neighbour.data.health -= 1
                neighbour.cur_reward += self.conf.lost_health_reward

                if neighbour.is_dead:
                    self.members[neighbour.data.team] -= 1
                    del self.world[visible.coord]

                    agent.cur_reward += self.conf.kill_reward
                else:
                    agent.cur_reward += self.conf.damage_reward
            else:
                raise ValueError('invalid action')

            agent.prev_reward = agent.cur_reward
            agent.cur_reward = 0.0

            agent.data.actions -= 1

        return self._done

    def reset(self):
        self.img.reset()
        self.world = SparsePlane()
        self.members = [len(self.agents[team]) for team in range(len(self.agents))]

        for team, agents in self.agents.items():
            for agent in agents:
                coord = None
                while coord is None or coord in self.world:
                    coord = tuple(random.randint(0, dim-1) for dim in self.conf.world_shape)

                self.world[coord] = agent
                agent.reset(
                    team=team,
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
                    data = self.world[i, j].data
                    color = self.img.color(group=data.team, alpha=data.health)
                else:
                    color = self.img.background

                field[i, j, :] = color

        self.img.update(field)
