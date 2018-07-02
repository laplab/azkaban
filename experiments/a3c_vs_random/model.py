import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class A3CModel(nn.Module):
    def __init__(self, in_units, n_actions, comm_shape):
        super(A3CModel, self).__init__()
        self.comm = np.zeros(comm_shape)

        self.fc1 = nn.Linear(in_units, 256)

        self.logits = nn.Linear(256, n_actions)
        self.state_value = nn.Linear(256, 1)

    def _get_state(self, obs):
        team, health, actions, *_ = obs

        middle = tuple(ax // 2 for ax in team.shape)
        current_team = team[middle]
        mask = (team == current_team)

        layers = mask, health, actions
        state = np.stack(layers, axis=-1).flatten().astype(float)

        tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        return tensor

    def forward(self, obs):
        x = self._get_state(obs)

        x = F.relu(self.fc1(x))

        logits = self.logits(x)
        state_value = self.state_value(x)

        return logits, state_value, self.comm
