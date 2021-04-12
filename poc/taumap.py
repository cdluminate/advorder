'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import pylab as lab
import numpy as np
import torch as th
from scipy.stats import kendalltau
import sys
from tqdm import tqdm
import argparse


def TauMap(AX, NC, *, attack=False):
    '''
    Let's see if we can find some heuristics for improving the kendall's tau
    '''
    if not attack:
        while True:
            candidates = th.rand(NC, 2) * AX
            query = th.rand(1, 2) * AX
            qcx = th.nn.functional.pairwise_distance(query, candidates, p=2)
            lab.scatter(candidates.numpy()[:, 1], candidates.numpy()[:, 0], c='red')
            lab.scatter(query.numpy()[:, 1], query.numpy()[:, 0], c='cyan')
            taumap = th.zeros(AX, AX)
            for i in tqdm(range(AX)):
                for j in range(AX):
                    qtmp = th.tensor([i, j])
                    qct = th.nn.functional.pairwise_distance(qtmp, candidates, p=2)
                    tau = kendalltau(qcx, qct).correlation
                    taumap[i, j] = tau
            lab.imshow(taumap)
            lab.colorbar()
            lab.show()
    else:
        while True:
            candidates = th.rand(NC, 2) * AX
            rperm = th.randperm(NC)
            lab.scatter(candidates.numpy()[:, 1], candidates.numpy()[:, 0], c='red')
            taumap = th.zeros(AX, AX)
            for i in tqdm(range(AX)):
                for j in range(AX):
                    qtmp = th.tensor([i, j])
                    qct = th.nn.functional.pairwise_distance(qtmp, candidates, p=2)
                    tau = kendalltau(rperm, qct).correlation
                    taumap[i, j] = tau
            lab.imshow(taumap, vmin=-1.0, vmax=1.0)
            lab.colorbar()
            lab.show()


if __name__ == '__main__':
    assert(kendalltau([0.1, 0.2, 0.3], [1, 2, 3]).correlation == 1.0)
    # Parse argument
    ag = argparse.ArgumentParser()
    ag.add_argument('-a', '--axis', type=int, default=96)
    ag.add_argument('-c', '--candidates', type=int, default=5)
    ag.add_argument('-t', '--atk', action='store_true')
    ag = ag.parse_args(sys.argv[1:])
    print(ag)
    # draw
    TauMap(ag.axis, ag.candidates, attack=ag.atk)
