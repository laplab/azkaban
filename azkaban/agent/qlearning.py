import random
from collections import defaultdict
from functools import partial

from azkaban.agent.core import Agent
import numpy as np


class TabularQLearningAgent(Agent):
    def __init__(self, conf, alpha, epsilon, discount, decay=None, shared_q=None):
        self.action_space = conf.action_space
        self.comm = np.zeros(conf.comm_shape)

        self.Q = shared_q or defaultdict(partial(defaultdict, int))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.decay = decay or 1.0

        self.prev_action = None
        self.prev_state = None

    def step(self, obs, reward, done):
        state = self._get_state(obs)

        if self.prev_action is not None:
            self._update(state, reward, done)

        action = self._get_action(state)
        self.prev_action = action
        self.prev_state = state

        self.epsilon *= self.decay

        return action, self.comm

    def reset(self):
        pass

    def _get_state(self, obs):
        team = obs.state.team
        state = []
        for visible in obs.view:
            if visible.team is not None:
                state.append(1 + (visible.team == team))
            else:
                state.append(0)

        return tuple(state)

    def _get_q(self, state, action):
        return self.Q[state][action]

    def _set_q(self, state, action, value):
        self.Q[state][action] = value

    def _get_best_action(self, state):
        best_action = None
        for action in self.action_space.iter():
            if best_action is None or self._get_q(state, action) > self._get_q(state, best_action):
                best_action = action

        return best_action

    def _get_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return self._get_best_action(state)

    def _update(self, next_state, reward, done):
        next_value = self._get_q(next_state, self._get_best_action(next_state))
        prev_value = self._get_q(self.prev_state, self.prev_action)
        reference_value = reward + self.discount * next_value * int(not done)

        updated_value = prev_value + self.alpha * (reference_value - prev_value)

        self._set_q(self.prev_state, self.prev_action, updated_value)
