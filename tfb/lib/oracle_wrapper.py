import numpy as np

from clamp_common_eval.defaults import get_test_oracle
import design_bench

from tqdm import tqdm


def get_oracle(args):
    if args.task == "tfbind":
        return TFBind8Wrapper(args)


class TFBind8Wrapper:
    def __init__(self, args):
        self.task = design_bench.make('TFBind8-Exact-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size]))
            scores += s.tolist()
        return np.array(scores)