from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F

from azkaban.agent.core import Agent
from azkaban.utils import NoOpContextManager


class AdvantageMode(Enum):
    ONESTEP = '1-step'
    NSTEP = 'n-step'
    GAE = 'gae'


class A3CParams(object):
    def __init__(self):
        self.gamma = 0.95
        self.lr = 1e-3
        self.entropy_coeff = 1e-2
        self.grad_max_norm = 50
        self.update_steps = 10
        self.advantage_mode = AdvantageMode.NSTEP
        self.value_loss_coeff = 0.25
        self.tau = 1.0


class A3CAgent(Agent):
    def __init__(self, conf, params, model, shared_model, shared_optimizer, trainable=True, lock=None):
        self.conf = conf

        self.params = params

        self.model = model
        self.shared_model = shared_model

        self.shared_optimizer = shared_optimizer

        self.trainable = trainable
        self.lock = lock or NoOpContextManager()

        self.rewards = None
        self.log_probs = None
        self.state_values = None
        self.entropies = None
        self.grad_updates = None

        self.prev_comm = None

        self.reset()

    def _ensure_shared_grads(self):
        for param, shared_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def _ensure_zero_grads(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

    def _clip_grads(self):
        if self.params.grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.grad_max_norm)

    def step(self, obs, prev_reward, comms, dcomm, done):
        # add reward for previous step if was any
        if len(self.state_values) > 0:
            self.rewards.append(torch.tensor(np.array([prev_reward]), dtype=torch.float32))

        if self.prev_comm is not None:
            self.grad_updates.append((self.prev_comm, dcomm))

        if (done or len(self.state_values) >= self.params.update_steps) and self.trainable:
            loss = torch.zeros((1, 1))

            if done:
                last_state_value = torch.zeros((1, 1))
            else:
                _, last_state_value, _, _ = self.model(obs, comms)

            self.state_values.append(last_state_value)

            n = len(self.state_values)

            reference_state_value = self.state_values[-1]
            gae = torch.zeros(1, 1)
            for t in range(n-2, -1, -1):
                cur_reward, cur_log_prob, cur_state_value = self.rewards[t], self.log_probs[t], self.state_values[t]
                cur_entropy, next_state_value = self.entropies[t], self.state_values[t+1]

                if self.params.advantage_mode == AdvantageMode.ONESTEP:
                    reference_state_value = cur_reward + self.params.gamma * next_state_value.detach()

                    temporal_difference = reference_state_value - cur_state_value
                    advantage = temporal_difference.detach()
                elif self.params.advantage_mode == AdvantageMode.NSTEP:
                    reference_state_value = cur_reward + self.params.gamma * reference_state_value.detach()

                    temporal_difference = reference_state_value - cur_state_value
                    advantage = temporal_difference.detach()
                elif self.params.advantage_mode == AdvantageMode.GAE:
                    reference_state_value = cur_reward + self.params.gamma * reference_state_value.detach()

                    temporal_difference = reference_state_value - cur_state_value

                    delta_t = cur_reward + self.params.gamma * next_state_value - cur_state_value
                    gae = gae * self.params.gamma * self.params.tau + delta_t

                    advantage = gae.detach()
                else:
                    raise ValueError('Unsupported advantage mode', self.params.advantage_mode)

                actor_pseudo_j = cur_log_prob * advantage + self.params.entropy_coeff * cur_entropy
                critic_loss = temporal_difference ** 2

                loss -= actor_pseudo_j.mean()
                loss += self.params.value_loss_coeff * critic_loss.mean()

            for a, b in self.grad_updates:
                a.backward(
                    gradient=torch.tensor(b, dtype=torch.float32).unsqueeze(0),
                    retain_graph=True,
                )

            loss.backward(retain_graph=True)
            # self._clip_grads()

            with self.lock:
                self._ensure_shared_grads()
                self.shared_optimizer.step()
                self.shared_optimizer.zero_grad()

            self._ensure_zero_grads()

            self.rewards = []
            self.log_probs = []
            self.state_values = []
            self.entropies = []
            self.grad_updates = []

            with self.lock:
                self.model.load_state_dict(self.shared_model.state_dict())

        # if self.prev_comm is not None:
        #     print('--- Before 2 ---')
        #     for parameter in self.model.parameters():
        #         mask = parameter.grad != 0
        #         if isinstance(mask, bool):
        #             print(mask)
        #         else:
        #             print((parameter.grad != 0).any())
        #     print('--- ----')

        logits, state_value, comm, dcomms = self.model(obs, comms)
        #
        # if self.prev_comm is not None:
        #     print('--- After ---')
        #     for parameter in self.model.parameters():
        #         mask = parameter.grad != 0
        #         if isinstance(mask, bool):
        #             print(mask)
        #         else:
        #             print((parameter.grad != 0).any())
        #     print('--- ----')

        self.prev_comm = comm

        dcomms = [dcomm.numpy() for dcomm in dcomms]

        state_value *= (1 - done)
        self.state_values.append(state_value)

        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        entropy = -(log_probs * probs).sum(1, keepdim=True)
        self.entropies.append(entropy)

        action = probs.multinomial(num_samples=1).data
        action_id = action.numpy()[0][0]

        log_prob = log_probs.gather(1, action)
        self.log_probs.append(log_prob)

        return action_id, np.asarray(comm.data).copy(), dcomms

    def reset(self):
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.entropies = []
        self.grad_updates = []
        self.prev_comm = None

        with self.lock:
            self.model.load_state_dict(self.shared_model.state_dict())
