#!/usr/bin/env python3
'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
from lib import snapshop as ss
import argparse
from termcolor import cprint, colored

ag = argparse.ArgumentParser()
ag.add_argument('-d', '--display', type=str, default='')
ag.add_argument('-a', '--anchor', type=str, default='')
ag.add_argument('-l', '--list', type=str, default='')
ag.add_argument('-k', '--topk', type=int, default=10)
ag.add_argument('-s', '--scrub', action='store_true')
ag = ag.parse_args()

if ag.display:
    ss.display(ag.display, k=ag.topk)
elif ag.anchor:
    ss.Anchor(ag.anchor, k=ag.topk)
elif ag.list:
    ss.detail(ag.list)
elif ag.scrub:
    from glob import glob
    initpkls = glob('__snapshop__/*init.pkl')
    scores = [1.0 for _ in initpkls]
    for (i, pkl) in enumerate(initpkls):
        try:
            score = ss.Anchor(pkl, k=ag.topk)
        except KeyError as e:
            score = 9.9
        scores[i] = score
    print('::: Summary', colored('GOOD', 'green'), colored('BAD', 'red'),
            colored('FAIR', 'yellow'))
    for (i, (pkl, score)) in enumerate(zip(initpkls, scores)):
        if score > 0.5:
            print("%4d"%i, colored("%.2f"%score, 'red'), pkl)
        elif score > 0.3:
            print("%4d"%i, colored("%.2f"%score, 'yellow'), pkl)
        else:
            print("%4d"%i, colored("%.2f"%score, 'green'), pkl)

else:
    print('???')
