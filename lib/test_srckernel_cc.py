'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import torch as th
from torch.utils.cpp_extension import load_inline
import reorder
import numpy as np
import random

#source='''
#torch::Tensor myfunc(torch::Tensor z) {
#	auto s = torch::ones_like(z);
#	return s + z;
#}
#'''
#
#mod = load_inline('mymod', source, functions=['myfunc'], verbose=True)
#x = th.rand(10)
#print(x)
#print(mod.myfunc(x))



with open("_srckernel.cc", 'rt') as f:
    srckernel = f.read()
SRC = load_inline('SRC', srckernel, functions=['ShortRangeRankingCorrelation'], verbose=True)

def BatchShortRangeRankingCorrelation(X, y, r):
    scores = np.zeros(X.shape[0])
    for (i, srt) in enumerate(X):
        scores[i] = SRC.ShortRangeRankingCorrelation(srt, y, r)
    return scores


import time
import rich
c = rich.get_console()

for i in range(100):
    for cansee in (5, 50, 1000):
        for k in (5, 10, 25):
            if k > cansee:
                continue
            #x = th.randint(1000, (cansee,))
            x = th.tensor(random.sample(range(1000), cansee))
            #y = th.randint(1000, (k,))
            y = th.tensor(random.sample(range(1000), k))
            r = th.randperm(k)
            s = SRC.ShortRangeRankingCorrelation(x, y, r)
            reference = reorder.NearsightRankCorr(x, y, r)
            if (s - reference > 1e-5):
                print(s - reference)
            assert(s - reference < 1e-5)

            c.print(i, cansee, k, 'OK')

x = th.randint(1000, (50,))
print('PY: x', x)
y = th.randint(1000, (50,))
print('PY: y', y)
r = th.randperm(5)
print('PY: r', r)
tm_start = time.time()
s = SRC.ShortRangeRankingCorrelation(x, y, r)
tm_one = time.time() - tm_start
reference = reorder.NearsightRankCorr(x, y, r)
c.print('  tm_one', tm_one)
c.print('  error', s - reference)

X = th.randint(1000, (50, 50))
tm_start = time.time()
S = BatchShortRangeRankingCorrelation(X, y, r)
tm_batch = time.time() - tm_start
reference = reorder.BatchNearsightRankCorr(X, y, r)
c.print('tm_batch', tm_batch)
c.print('  error', S - reference)

