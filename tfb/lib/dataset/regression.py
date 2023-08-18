import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from clamp_common_eval.defaults import get_default_data_splits
import design_bench
import pandas as pd
import os.path as osp

from lib.dataset.base import Dataset


class TFBind8Dataset(Dataset):
    def __init__(self, args, oracle):
        super().__init__(args, oracle)
        self._load_dataset()
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self):
        task = design_bench.make('TFBind8-Exact-v0')
        x = task.x
        y = task.y.reshape(-1)
        self.train, self.valid, self.train_scores, self.valid_scores = train_test_split(x, y, test_size=0.1, random_state=self.rng)

    def create_all_stochastic_datasets(self, stick):
        self.train_thought = self.create_stochastic_data(stick, self.train)
        self.valid_thought = self.create_stochastic_data(stick, self.valid)
        print ('\033[32mfinished creating stochastic datasets for train, valid\033[0m')
        
    def create_stochastic_data(self, stick, det_data):
        stochastic_data = []
        for idx in range(len(det_data)):
            curr_seq = det_data[idx]
            curr_len = len(curr_seq)
            
            curr_rand_probs = np.random.rand(curr_len)
            curr_rand_actions = np.random.randint(low=0, high=4, size=curr_len)

            curr_actions = []
            for step_idx in range(curr_len):
                if curr_rand_probs[step_idx] < stick:
                    # take a random action
                    curr_actions.append(curr_rand_actions[step_idx])
                else:
                    # just take this action
                    curr_actions.append(curr_seq[step_idx])

            stochastic_data.append(curr_actions)
        return stochastic_data

    def sample(self, n):
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices], [self.train_scores[i] for i in indices])
    
    def sample_with_stochastic_data(self, n):
        indices = np.random.randint(0, len(self.train), n)
        return ([[self.train[i], self.train_thought[i]] for i in indices], [self.train_scores[i] for i in indices])

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        train_seq, val_seq = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/10):
                val_seq.append(x)
                val.append(score)
            else:
                train_seq.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)
        self.train = np.concatenate((self.train, train_seq), axis=0)
        self.valid = np.concatenate((self.valid, val_seq), axis=0)
    
    def _tostr(self, seqs):
        return ["".join([str(i) for i in x]) for x in seqs]

    def _top_k(self, data, k):
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.array(data[1])[indices]
        topk_prots = np.array(data[0])[indices]

        return self._tostr(topk_prots), topk_scores

    def top_k(self, k):
        data = (np.concatenate((self.train, self.valid), axis=0), np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]), axis=0)

        data = (seqs, scores)
        
        return self._top_k(data, k)


