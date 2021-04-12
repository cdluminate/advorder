'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.

Black-box Order Attack Implementation
'''
from random import random
from scipy.stats import kendalltau
from termcolor import cprint, colored
from tqdm import tqdm
from typing import *
import math
import numpy as np
import os
import sys
import torch as th
import yaml
from time import time
from multiprocessing.dummy import Pool
from joblib import Parallel, delayed


def BatchNearsightRankCorr(argsort, otopk, rperm, *, debug=False) -> np.ndarray:
    scores = np.zeros(argsort.shape[0])
    # [serial] -- moderate
    for (i, srt) in enumerate(argsort):
        scores[i] = NearsightRankCorr(srt, otopk, rperm, debug=debug)
    # [multithread]
    #with Pool(4) as p:
    #    scores[:] = list(map(lambda x: NearsightRankCorr(x, otopk, rperm, debug=debug), argsort))
    # [joblib] -- very slow
    #scores[:] = list(Parallel(n_jobs=2)(
    #    delayed(lambda x: NearsightRankCorr(x, otopk, rperm, debug=debug))(y)
    #    for y in argsort))
    return scores


def NearsightRankCorr(
        argsort: th.Tensor,  # realtime top-k ranking result (topN) list[idx]
        otopk: th.Tensor,    # original top-k ranking result (topk) list[idx]
        rperm: th.Tensor,    # desired permutation of the topk results list[perm]
        *,
        debug=False,
        mode='numpy',
        ) -> float:
    '''
    Calculate the score matrix for the evolutionary algorithm
    argsort is the partial decision (canseek)
    rtopk is the specified permutation for the original topk candidates

    performance: 150 it/s (numpy mode) ; 110it/s (torch mode)

    # test: fully concordant
    >>> NearsightRankCorr(th.arange(5)+20, th.arange(5)+20, th.arange(5))
    1.0
    >>> NearsightRankCorr(th.arange(5)+20, th.arange(5)+20, th.tensor([4,3,2,1,0]))
    -1.0
    >>> NearsightRankCorr(th.arange(20)+20, th.arange(5)+20, th.arange(5))
    1.0
    >>> NearsightRankCorr(th.arange(20)+20, th.arange(5)+20, th.tensor([4,3,2,1,0]))
    -1.0
    '''
    if mode == 'torch':
        argsort = argsort.detach().cpu().flatten()
        otopk = otopk.detach().cpu().flatten()
        rtopk = otopk[rperm.cpu()].flatten()
        assert(len(argsort) > len(rtopk))
        scores = th.zeros(len(rperm), len(rperm))  # TriL
        for i in range(len(rperm)):
            for j in range(len(rperm)):
                cranki = th.nonzero(argsort == otopk[i], as_tuple=True)[0][0]
                crankj = th.nonzero(argsort == otopk[j], as_tuple=True)[0][0]
                xranki = th.nonzero(rtopk == otopk[i], as_tuple=True)[0][0]
                xrankj = th.nonzero(rtopk == otopk[j], as_tuple=True)[0][0]
                if (cranki>crankj and xranki>xrankj) or (cranki<crankj and xranki<xrankj):
                    scores[i, j] = 1
                elif (cranki>crankj and xranki<xrankj) or (cranki<crankj and xranki>xrankj):
                    scores[i, j] = -1
                else:
                    scores[i, j] = -1
        score = scores.tril(diagonal=-1).sum() / ((len(rperm) * (len(rperm)-1))/2)
        return score
    # Casting from Torch to Numpy
    argsort = argsort.detach().cpu().numpy().flatten()
    otopk = otopk.detach().cpu().numpy().flatten()
    rtopk = otopk[rperm.cpu()].flatten()
    if len(argsort) < len(rperm):
        print('len(argsort)', len(argsort), argsort.shape)
        print('len(rperm)', len(rperm), rperm.shape)
        raise ValueError(f'invalid argsort and rperm')
    if debug:
        print('1. argsort', argsort)
        print('2. otopk', otopk)
        print('3. rperm', rperm)
    # Generate the scores (tril) matrix
    scores = np.zeros((len(rperm), len(rperm)))  # triL
    for i in range(len(rperm)):
        if otopk[i] not in argsort:
            scores[i, :] = -1
            continue
        #if not np.where(argsort == otopk[i])[0]: continue
        for j in range(i):
            if otopk[j] not in argsort:
                scores[:, j] = -1
                continue
            #if not np.where(argsort == otopk[j])[0]: continue  # slow
            cranki = np.where(argsort == otopk[i])[0][0]
            crankj = np.where(argsort == otopk[j])[0][0]
            xranki = np.where(rtopk == otopk[i])[0][0]
            xrankj = np.where(rtopk == otopk[j])[0][0]
            # case 1: concordant
            if (cranki > crankj and xranki > xrankj) \
                    or (cranki < crankj and xranki < xrankj):
                scores[i, j] = 1
            # case 2: discordant
            elif (cranki > crankj and xranki < xrankj) \
                    or (cranki < crankj and xranki > xrankj):
                scores[i, j] = -1
            else:
                pass # score=0
    # semantics trimming is automatically satisfied.
    score = np.tril(scores, k=-1).sum() / ((len(rperm) * (len(rperm)-1))//2)
    if debug:
        print('4. scores\n', scores)
        print('5. score', score)
    return score
