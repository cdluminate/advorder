'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.

Reference:
https://stackoverflow.com/questions/5862915/passing-numpy-arrays-to-a-c-function-for-input-and-output
'''
import os
import ctypes
import multiprocessing.dummy as mp
import numpy as np
os.putenv('LD_LIBRARY_PATH', '.:./lib/')

BUILDCMD = '''
cd lib/srck;
cargo build --release;
cp -av target/release/libsrck.so ../../;
cargo clean;
'''
if not os.path.exists('libsrck.so'):
    os.system(BUILDCMD)
else:
    # recompile every time.
    os.system(BUILDCMD)

try:
    libsrck = ctypes.CDLL('./libsrck.so')
except OSError:
    libsrck = ctypes.CDLL('./lib/libsrck.so')

class array(ctypes.Structure):
    _fields_ = [
            ('data', ctypes.c_void_p),
            ('len', ctypes.c_int32)
            ]
    @staticmethod
    def From(x: np.ndarray):
        buf = ctypes.c_void_p(x.ctypes.data)
        return array(buf, len(x))

libsrck.print_array.argtypes = (array,)
libsrck.print_array.restype = None

libsrck.ShortrangeRankingCorrelation.argtypes = (array, array, array)
libsrck.ShortrangeRankingCorrelation.restype = ctypes.c_float

def NearsightRankCorr(x, y, r):
    x = x.cpu().clone().detach().numpy().astype(np.int32)
    y = y.cpu().clone().detach().numpy().astype(np.int32)
    r = r.cpu().clone().detach().numpy().astype(np.int32)
    return libsrck.ShortrangeRankingCorrelation(
            *(array.From(X) for X in [x, y, r]))

def BatchNearsightRankCorr(X, y, r):
    # [method 2]: thread pool
    #with mp.Pool(4) as pool:
    #    L = list(pool.map(lambda z: NearsightRankCorr(z, y, r), X))
    #return np.array(L)
    # [method 1]: serial
    return np.array([NearsightRankCorr(x, y, r) for x in X])


if __name__ == '__main__':
    x = (np.random.rand(10) * 10).astype(np.int32)
    print(x)
    libsrck.print_array(array.From(x))

    import reorder
    import time
    import rich
    import torch as th
    import random
    c = rich.get_console()

    if True:
        time_start = time.time()
        cansee, k = 50, 25
        for i in range(1000):
            x = np.array(random.sample(range(1000), cansee),
                    dtype=np.int32).ravel()
            y = np.array(random.sample(range(1000), k),
                    dtype=np.int32).ravel()
            r = th.randperm(k).numpy().astype(np.int32)
            s = libsrck.ShortrangeRankingCorrelation(
                    *(array.From(X) for X in [x, y, r]))
        time_end = time.time()
        print('K=25, 1000 times, elapsed', time_end - time_start)

    if False:
        exit(1)
    for i in range(100):
        for cansee in (5, 50, 1000):
            for k in (5, 10, 25):
                if k > cansee:
                    continue
                #x = th.randint(1000, (cansee,))
                x = np.array(random.sample(range(1000), cansee),
                        dtype=np.int32).ravel()
                #print('(py)argsort=', x)
                #libsrck.print_array(array.From(x))
                #y = th.randint(1000, (k,))
                y = np.array(random.sample(range(1000), k),
                        dtype=np.int32).ravel()
                #print('(py)otopk=', y)
                #libsrck.print_array(array.From(y))
                r = th.randperm(k).numpy().astype(np.int32)
                #print('(py)rperm=', r)
                #libsrck.print_array(array.From(r))
                s = libsrck.ShortrangeRankingCorrelation(
                        *(array.From(X) for X in [x, y, r]))
                ss = NearsightRankCorr(
                        *(th.from_numpy(X) for X in [x, y, r]))
                reference = reorder.NearsightRankCorr(
                        *(th.from_numpy(X) for X in [x, y, r]))
                print(s, ss, reference, s - reference)
                assert(s == ss)
                if (s - reference > 1e-5):
                    print(s, ss, reference, s - reference)
                assert(s - reference < 1e-5)

                c.print(i, cansee, k, 'OK')

