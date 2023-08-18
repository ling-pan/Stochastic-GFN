import torch
import torch.nn.functional as F
import torch.nn as nn

from lib.generator.base import GeneratorBase
from lib.model.mlp import MLP

import numpy as np
from itertools import chain

import itertools
from torch.distributions import Categorical
from tqdm import tqdm

import h5py
import time

from sklearn import manifold

class DBGFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.num_tokens = args.vocab_size
        self.max_len = args.gen_max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        
        self.stick = args.stick
        num_outputs = self.num_tokens + self.num_tokens + 1
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=num_outputs, 
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))
        self.device = args.device
        
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()
        return loss, info

    def get_loss(self, batch):
        strs, thought_strs, r = zip(*batch["bulk_trajs"])

        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens) 
        inp[:, :inp_x.shape[1], :] = inp_x
        
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        lens = [len(i) for i in strs]

        model_outs = self.model(x, None, return_all=True, lens=lens) 
        pol_logits = model_outs[:, :, :self.num_tokens] 
        pol_back_logits = model_outs[:, :, self.num_tokens:-1] 
        log_flows = model_outs[:, :, -1] 

        pol_logits = self.logsoftmax2(pol_logits)[:-1] 
        pol_back_logits = self.logsoftmax2(pol_back_logits)[1:] 
    
        mask = s.eq(self.num_tokens)

        s = s.swapaxes(0, 1) 
        thought_s = thought_s.swapaxes(0, 1)

        n = (s.shape[0] - 1) * s.shape[1]

        pol_logits = pol_logits.reshape((n, self.num_tokens)) 
        pol_logits = pol_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_logits = pol_logits.reshape(s[1:].shape) 
        pol_back_logits = pol_back_logits.reshape((n, self.num_tokens))
        pol_back_logits = pol_back_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_back_logits = pol_back_logits.reshape(s[1:].shape)

        mask = mask[:, 1:].swapaxes(0, 1).logical_not().float() 

        ll_diff = torch.zeros((pol_logits.shape)).to(self.device) 
        ll_diff += log_flows[:-1] 
        ll_diff += pol_logits
        log_flows = log_flows[1:].transpose(1, 0) 
        r = r.clamp(min=self.reward_exp_min).log()
        r = r.unsqueeze(-1).repeat(1, log_flows.shape[1]) 
        lens = torch.tensor(lens).long()
        end_pos = lens - 1 - 1
        mask_for_backward = mask.clone().detach().transpose(1, 0) 
        mask_for_backward[torch.arange(end_pos.shape[0], device=self.device), end_pos] -= 1
        end_log_flow = mask_for_backward * log_flows + (1 - mask_for_backward) * r
        end_log_flow = end_log_flow.transpose(1, 0)
        ll_diff -= end_log_flow

        ll_diff -= pol_back_logits
        ll_diff *= mask
        loss = (ll_diff ** 2).sum() / mask.sum()
        info = {'gfn_loss': loss.item()}

        return loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        assert not return_all

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out    


class ReplayBuffer(object):
    def __init__(self, max_len, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.strs = np.zeros((max_size, max_len), dtype=int)
        self.thought_strs = np.zeros((max_size, max_len), dtype=int)
        self.rewards = np.zeros((max_size,))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, strs, thought_strs, rews):
        for i in range(len(strs)):
            curr_str, curr_thought_str, curr_rew = strs[i], thought_strs[i], rews[i]
            self.strs[self.ptr] = curr_str
            self.thought_strs[self.ptr] = curr_thought_str
            self.rewards[self.ptr] = curr_rew

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        sampled_strs = self.strs[ind]
        sampled_thought_strs = self.thought_strs[ind]
        sampled_rs = self.rewards[ind]

        return sampled_strs, sampled_thought_strs, sampled_rs
       

class StochasticDBGFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.num_tokens = args.vocab_size
        self.max_len = args.gen_max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        
        self.stick = args.stick
        num_outputs = self.num_tokens + self.num_tokens + 1
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=num_outputs, 
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))
        self.device = args.device
        
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

        self.forward_dynamics = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=self.num_tokens, 
            num_hid=args.dynamics_num_hid,
            num_layers=args.dynamics_num_layers,
            max_len=self.max_len + 1,
            dropout=0,
            partition_init=args.gen_partition_init, 
            causal=args.gen_do_explicit_Z 
        )
        print (self.forward_dynamics)
        self.forward_dynamics.to(args.device)

        self.dynamics_opt = torch.optim.Adam(self.forward_dynamics.model_params(), args.dynamics_lr, weight_decay=args.dynamics_L2, betas=(0.9, 0.999))
        self.dynamics_clip = args.dynamics_clip

        self.ce_loss = nn.CrossEntropyLoss()

        self.dynamics_off_pol = args.dynamics_off_pol
        if self.dynamics_off_pol:
            self.dynamics_buffer = ReplayBuffer(self.max_len)
            self.dynamics_sample_size = args.dynamics_sample_size
            self.dynamics_off_pol_rounds = args.dynamics_off_pol_rounds

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()

        rets = [loss.item()]

        if self.dynamics_off_pol:
            total_dynamics_loss = 0.
            for dynamics_off_pol_round in range(self.dynamics_off_pol_rounds):
                dynamics_loss = self.get_dynamics_loss()
                dynamics_loss.backward()
                if self.dynamics_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
                self.dynamics_opt.step()
                self.dynamics_opt.zero_grad()
                total_dynamics_loss += dynamics_loss.item()
            dynamics_loss = total_dynamics_loss / self.dynamics_off_pol_rounds
        else:
            dynamics_loss = info['forward_dynamics_loss']
            dynamics_loss.backward()
            if self.dynamics_clip > 0.:
                torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
            self.dynamics_opt.step()
            self.dynamics_opt.zero_grad()
            dynamics_loss = dynamics_loss.item()

        rets.append(dynamics_loss)

        return rets

    def get_dynamics_loss(self):
        info = {}
        strs, thought_strs, r = self.dynamics_buffer.sample(self.dynamics_sample_size)

        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        lens = [len(i) for i in strs]

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
        inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
        x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()

        real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)

        forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
        forward_model_outs = forward_model_outs[:-1, :, :]

        forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))

        forward_model_logits = forward_model_outs.detach().log_softmax(-1)

        return forward_dynamics_loss

    def get_loss(self, batch):
        info = {}
        strs, thought_strs, r = zip(*batch["bulk_trajs"])

        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        lens = [len(i) for i in strs]

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
        inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
        x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()

        real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)

        if not self.dynamics_off_pol:
            forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
            forward_model_outs = forward_model_outs[:-1, :, :]

            forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))

            forward_model_logits = forward_model_outs.detach().log_softmax(-1)
        else:
            with torch.no_grad():
                forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
            forward_model_outs = forward_model_outs[:-1, :, :]

            forward_dynamics_loss = 1e9

            forward_model_logits = forward_model_outs.log_softmax(-1)
        
        forward_model_logits = forward_model_logits.gather(-1, real_actions.unsqueeze(-1)).squeeze(-1) 

        info['forward_dynamics_loss'] = forward_dynamics_loss

        model_outs = self.model(x, None, return_all=True, lens=lens) 
        pol_logits = model_outs[:, :, :self.num_tokens] 
        pol_back_logits = model_outs[:, :, self.num_tokens:-1] 
        log_flows = model_outs[:, :, -1] 

        pol_logits = self.logsoftmax2(pol_logits)[:-1] 
        pol_back_logits = self.logsoftmax2(pol_back_logits)[1:] 

        mask = s.eq(self.num_tokens)

        s = s.swapaxes(0, 1)
        thought_s = thought_s.swapaxes(0, 1)

        n = (s.shape[0] - 1) * s.shape[1]

        pol_logits = pol_logits.reshape((n, self.num_tokens))
        pol_logits = pol_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_logits = pol_logits.reshape(s[1:].shape)
        pol_back_logits = pol_back_logits.reshape((n, self.num_tokens))
        pol_back_logits = pol_back_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_back_logits = pol_back_logits.reshape(s[1:].shape)

        mask = mask[:, 1:].swapaxes(0, 1).logical_not().float() 

        ll_diff = torch.zeros((pol_logits.shape)).to(self.device)
        ll_diff += log_flows[:-1]
        ll_diff += pol_logits
        ll_diff += forward_model_logits
        log_flows = log_flows[1:].transpose(1, 0) 
        r = r.clamp(min=self.reward_exp_min).log()
        r = r.unsqueeze(-1).repeat(1, log_flows.shape[1]) 
        lens = torch.tensor(lens).long()
        end_pos = lens - 1 - 1
        mask_for_backward = mask.clone().detach().transpose(1, 0) 
        mask_for_backward[torch.arange(end_pos.shape[0], device=self.device), end_pos] -= 1
        end_log_flow = mask_for_backward * log_flows + (1 - mask_for_backward) * r
        end_log_flow = end_log_flow.transpose(1, 0)
        ll_diff -= end_log_flow
        ll_diff -= pol_back_logits
        ll_diff *= mask
        loss = (ll_diff ** 2).sum() / mask.sum()

        return loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        assert not return_all

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out    

