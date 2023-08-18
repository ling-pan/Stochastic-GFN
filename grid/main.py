import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import wandb
import random
import sys
import tempfile
import datetime
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--progress", action='store_true')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--dir", default='./results', type=str)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--lr", default=0.001, help="Learning rate", type=float)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=20000, type=int)
parser.add_argument("--dynamics_lr", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--dynamics_hid_dim", default=256, type=int)
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--ndim", default=2, type=int)
parser.add_argument("--stick", default=0.25, type=float) 

_dev = [torch.device('cuda')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

def set_device(dev):
    _dev[0] = dev

def func_corners(x, kind=None):
    ax = abs(x)
    r = (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-3
    return r

def sticky_based_action_modification(states, acts, stick, dev, action_dim, horizon):
    action_dim -= 1
    rand_probs = np.random.rand(acts.shape[0])
    flags = torch.tensor(rand_probs < stick).long().to(dev)
    flags1 = (acts != action_dim).long()
    flags *= flags1
    flags2 = (states == horizon - 1).sum(-1)
    flags *= (1 - flags2)
    rand_acs = torch.tensor(np.random.randint(0, action_dim, rand_probs.shape)).long().to(dev)
    sticky_acs = acts * (1 - flags) + rand_acs * flags
    return sticky_acs

class GridEnv:
    def __init__(self, horizon, ndim=2, xrange=[-1, 1], func=None):
        self.horizon = horizon
        self.ndim = ndim
        self.func = func
        self.xspace = np.linspace(*xrange, horizon) 
        rs = []
        for i in range(self.horizon):
            for j in range(self.horizon):
                rs.append(self.func(self.s2x(np.int32([i, j]))))
        rs = np.array(rs)
        self.true_density = rs / rs.sum()

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z

    def s2x(self, s):
        x = (self.obs(s).reshape((self.ndim, self.horizon)) * self.xspace[None, :]).sum(1)
        return x

    def reset(self):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        rew = self.func(self.s2x(self._state))
        return self.obs(), rew, self._state

    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1
        done = (a == self.ndim)
        if _s is None:
            self._state = s
            self._step += 1
        rew = 0 if not done else self.func(self.s2x(s))
        return self.obs(s), rew, done, s

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    return nn.Sequential(*(sum([[nn.Linear(i, o)] + ([act] if n < len(l)-2 else []) for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, children_state_dim, dev):
        super(TransitionModel, self).__init__()
        self.transition_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, children_state_dim)
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.dev = dev
        self.action_dim = action_dim

    def forward(self, states, actions):
        a_reps = F.one_hot(actions, num_classes=self.action_dim).float()
        a_reps = a_reps.squeeze(1)
        sas = torch.cat((states, a_reps), dim=-1) 
        s_next_outs = self.transition_model(sas) 
        return s_next_outs

    def convert_s_labels(self, states, transitted_states):
        diff = transitted_states - states
        indices = diff.nonzero()
        reverted_as = (torch.ones(states.shape[0]) * 2).long().to(self.dev)
        for idx in range(indices.shape[0]):
            curr_row_idx, curr_ac = indices[idx]
            reverted_as[curr_row_idx] = curr_ac
        return reverted_as

    def compute_loss(self, states, actions, states_coor_form, s_labels_raw):
        s_labels_converted = self.convert_s_labels(states_coor_form, s_labels_raw)
        predicted_s_outs = self.forward(states, actions)
        loss = self.ce_loss(predicted_s_outs, s_labels_converted.long())
        return loss

class StochasticGFN:
    def __init__(self, args, envs):
        assert args.ndim == 2 and args.stick > 0.

        out_dim = args.ndim + 1 + args.ndim * args.ndim + 1
        self.model = make_mlp([args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [out_dim])
        self.model.to(args.dev)

        self.forward_dynamics = TransitionModel(args.horizon * args.ndim, args.ndim + 1, args.dynamics_hid_dim, args.ndim + 1, args.dev)
        self.forward_dynamics.to(args.dev)

        self.stick = args.stick
        self.envs = envs
        self.ndim = args.ndim
        self.horizon = args.horizon
        
        self.dev = args.dev

    def parameters(self):
        return self.model.parameters()

    def transition_models_parameters(self):
        return self.forward_dynamics.parameters()

    def sample_many(self, mbsize, all_visited):
        inf = 1000000000

        batch_s, batch_a = [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize

        while not all(done):
            with torch.no_grad():
                pred = self.model(s)

                z = torch.where(s > 0)[1].reshape(s.shape[0], -1)
                z[:, 1] -= self.horizon

                edge_mask = torch.cat([(z == self.horizon - 1).float(), torch.zeros((len(done) - sum(done), 1), device=args.dev)], 1)
                logits = (pred[..., : args.ndim + 1] - inf * edge_mask).log_softmax(1)

                sample_ins_probs = logits.softmax(1)
                acts = sample_ins_probs.multinomial(1)
                acts = acts.squeeze(-1)

            formatted_s = torch.where(s == 1)[1].reshape(-1, 2)
            formatted_s[:, 1] -= self.horizon
            noisy_acts = sticky_based_action_modification(formatted_s, acts, self.stick, self.dev, self.ndim + 1, args.horizon)

            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], noisy_acts)]

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, acts)):
                env_idx = not_done_envs[dat_idx]

                curr_formatted_s = torch.where(curr_s > 0)[0]
                curr_formatted_s[1] -= self.horizon

                batch_s[env_idx].append(curr_formatted_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

            for dat_idx, (ns, r, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    env_idx_return_map[env_idx] = r.item()

                    formatted_ns = np.where(ns > 0)[0]
                    formatted_ns[1] -= self.horizon
                    formatted_ns = formatted_ns.tolist()

                    batch_s[env_idx].append(tl(formatted_ns))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(sp[0] * self.horizon + sp[1])

        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])

            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1

        batch_R = []
        for i in range(len(batch_s)):
            batch_R.append(env_idx_return_map[i])

        return [batch_s, batch_a, batch_R, batch_steps]

    def convert_states_to_onehot(self, states):
        return torch.nn.functional.one_hot(states, self.horizon).view(states.shape[0], -1).float()

    def learn_dynamics_models(self, batch):
        states, actions, returns, episode_lens = batch
        forward_losses = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][:curr_episode_len, :] 
            curr_actions = actions[data_idx][:curr_episode_len - 1, :] 
            curr_return = returns[data_idx]

            curr_states_onehot = self.convert_states_to_onehot(curr_states)

            forward_loss = self.forward_dynamics.compute_loss(curr_states_onehot[:-1], curr_actions, curr_states[:-1], curr_states[1:])
            forward_losses.append(forward_loss)

        forward_losses = torch.stack(forward_losses)
        loss = forward_losses.sum() / len(forward_losses)
        return loss

    def convert_sa_labels(self, states, actions, next_states):
        diff = next_states - states
        indices = diff.nonzero()[:, 1]
        sa_idxs = []
        for idx in range(indices.shape[0]):
            if indices[idx] == 0 and actions[idx] == 0:
                sa_idxs.append(0)
            elif indices[idx] == 1 and actions[idx] == 0:
                sa_idxs.append(1)
            elif indices[idx] == 0 and actions[idx] == 1:
                sa_idxs.append(2)
            elif indices[idx] == 1 and actions[idx] == 1:
                sa_idxs.append(3)
        sa_idxs = torch.tensor(sa_idxs).long().to(self.dev)
        return sa_idxs

    def learn_from(self, it, batch):
        inf = 1000000000

        states, actions, returns, episode_lens = batch
        returns = torch.tensor(returns).to(self.dev)

        ll_diff = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][:curr_episode_len, :] 
            curr_actions = actions[data_idx][:curr_episode_len - 1, :]
            curr_return = returns[data_idx]

            curr_states_onehot = self.convert_states_to_onehot(curr_states)

            transitted_s_idxs = self.forward_dynamics.convert_s_labels(curr_states[:-1], curr_states[1:]).unsqueeze(-1) 
            with torch.no_grad():
                forward_model_outs = self.forward_dynamics(curr_states_onehot[:-1], curr_actions)
            forward_model_logits = forward_model_outs.log_softmax(1)
            forward_model_logits = forward_model_logits.gather(1, transitted_s_idxs).squeeze(1) 

            pred = self.model(curr_states_onehot)

            edge_mask = torch.cat([(curr_states == self.horizon - 1).float(), torch.zeros((curr_states.shape[0], 1), device=self.dev)], 1)
            logits = (pred[..., :self.ndim + 1] - inf * edge_mask).log_softmax(1) 

            init_edge_mask = (curr_states == 0).float() 
            init_edge_mask = init_edge_mask.repeat(1, self.ndim) 
            backward_model_logits = (pred[..., self.ndim + 1:-1] - inf * init_edge_mask).log_softmax(1)
            back_transitted_sa_idxs = self.convert_sa_labels(states=curr_states[:-2], actions=curr_actions[:-1], next_states=curr_states[1:-1]).unsqueeze(-1) 
            backward_model_logits = backward_model_logits[1:-1].gather(1, back_transitted_sa_idxs).squeeze(1) 
            
            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1) 
            log_flow = pred[..., -1][:-1] 

            curr_ll_diff = torch.zeros(curr_states.shape[0] - 1).to(self.dev)
            curr_ll_diff += log_flow
            curr_ll_diff += logits
            curr_ll_diff += forward_model_logits
            curr_ll_diff[:-1] -= log_flow[1:]
            curr_ll_diff[:-1] -= backward_model_logits
            curr_ll_diff[-1] -= curr_return.float().log()

            ll_diff.append(curr_ll_diff ** 2)

        ll_diff = torch.cat(ll_diff)
        loss = ll_diff.sum() / len(ll_diff)

        return loss

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    args.dev = torch.device(args.device)
    set_device(args.dev)

    envs = [GridEnv(args.horizon, args.ndim, func=func_corners) for i in range(args.mbsize)]

    agent = StochasticGFN(args, envs)

    opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.lr}])
    dynamics_opt = torch.optim.Adam([{'params': agent.transition_models_parameters(), 'lr': args.dynamics_lr}])

    all_visited = []
    for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):
        data = agent.sample_many(args.mbsize, all_visited) 

        dynamics_loss = agent.learn_dynamics_models(data)
        dynamics_loss.backward()
        dynamics_opt.step()
        dynamics_opt.zero_grad()

        loss = agent.learn_from(i, data) 
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 10 == 0:
            emp_dist = np.bincount(all_visited[-200000:], minlength=len(envs[0].true_density)).astype(float)
            emp_dist /= emp_dist.sum()
            l1 = np.abs(envs[0].true_density - emp_dist).mean()
            print (i, l1)
            
if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
