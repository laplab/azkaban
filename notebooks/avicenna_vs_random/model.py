import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class AvicennaModel(nn.Module):
    def __init__(self, in_units, n_actions, comm_size):
        super(AvicennaModel, self).__init__()

        self.fc1 = nn.Linear(in_units, 256)

        self.logits = nn.Linear(256, n_actions)
        self.state_value = nn.Linear(256, 1)
        self.comm = nn.Linear(256, comm_size)

    def _get_state(self, obs):
        team, health, actions, comm, comm_grad = obs

        # agent is in the center - get his team
        middle = tuple(ax // 2 for ax in team.shape)
        current_team = team[middle]

        # create a mask where
        # 0 - empty cell
        # 1 - buddy
        # 2 - enemy
        mask = (team == current_team) + 1

        # stack layers over the third dimension and flatten
        layers = mask, health
        state = np.stack(layers, axis=-1).flatten()

        # sum all communication vectors into one "packed" vector
        packed_comm = comm.sum(axis=0).sum(axis=0)

        # stack two vectors one to another to create a view
        # view = np.concatenate((state, packed_comm), axis=0)
        view = state

        # add batch dimension
        batched_view = np.expand_dims(view, axis=0).astype(float)

        tensor = torch.tensor(batched_view, dtype=torch.float32)
        return tensor

    def forward(self, obs):
        x = self._get_state(obs)

        x = F.relu(self.fc1(x))

        logits = self.logits(x)
        state_value = self.state_value(x)
        comm = self.comm(x)

        return logits, state_value, comm
