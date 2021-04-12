'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import torch as th
from torch.utils.cpp_extension import load_inline
import numpy as np
import os

# NOTE: inspect the current parallel backend with
# print(th.__config__.parallel_info())
# ref: https://github.com/suphoff/pytorch_parallel_extension_cpp/blob/master/setup.py

flags = ['-DAT_PARALLEL_OPENMP', '-fopenmp']
# flags = ['-DAT_PARALLEL_NATIVE_TBB']
# flags = ['-DAT_PARALLEL_NATIVE']

__srcpath = os.path.join(os.path.dirname(__file__), '_srckernel.cc')
with open(__srcpath, 'rt') as f:
    srckernel = f.read()
SRC = load_inline('SRC', srckernel, functions=[
    'ShortRangeRankingCorrelation',
    'BatchShortRangeRankingCorrelation',
    ],
    extra_cflags=flags + ['-O2'],
    extra_ldflags=flags,
    verbose=True)

def BatchNearsightRankCorr(X, y, r):
    X = X.cpu()
    y = y.cpu()
    r = r.cpu()
    scores = SRC.BatchShortRangeRankingCorrelation(X, y, r)
    return scores.cpu().numpy().astype(np.float)

def NearsightRankCorr(x, y, r):
    x = x.cpu()
    y = y.cpu()
    r = r.cpu()
    return SRC.ShortRangeRankingCorrelation(x, y, r)
