import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class AvicennaModel(nn.Module):
    def __init__(self, obs_size, comm_size, n_actions):
        super(AvicennaModel, self).__init__()

        self.comm_size = comm_size
        self.fc_obs = nn.Linear(obs_size, 240)
        self.fc_comm = nn.Linear(comm_size, 16)

        self.logits = nn.Linear(256, n_actions)
        self.state_value = nn.Linear(256, 1)
        self.comm = nn.Linear(256, comm_size)
        self.bn = nn.BatchNorm1d(num_features=comm_size)

    def _vectorize_obs(self, obs):
        team, health, actions = obs

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
        view = np.stack(layers, axis=-1).flatten()

        tensor = torch.tensor(view, dtype=torch.float32)
        return tensor

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def forward(self, obs, comms):
        obs = self._vectorize_obs(obs)
        obs = F.relu(self.fc_obs(obs))

        packed_comm = torch.zeros(self.comm_size)
        comm_tensors = []
        for comm in comms:
            tensor = torch.tensor(comm, dtype=torch.float32, requires_grad=True)
            comm_tensors.append(tensor)
            packed_comm += tensor
        packed_comm = F.relu(self.fc_comm(packed_comm))

        x = torch.cat((obs, packed_comm))
        x = x.unsqueeze(0)

        logits = self.logits(x)
        state_value = self.state_value(x)
        comm = self.comm(x)
        comm = F.sigmoid(comm)

        dcomms = torch.autograd.grad(state_value, comm_tensors, retain_graph=True)
        self.zero_grad()

        return logits, state_value, comm, dcomms
