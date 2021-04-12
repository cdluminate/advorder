#!/usr/bin/env python3
'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''

from lib import snapshop as ss
import argparse
import pickle

if __name__ == '__main__':

    ag = argparse.ArgumentParser()
    ag.add_argument('-f', '--file', help='pickle file to load', required=True)
    ag = ag.parse_args()

    with open(ag.file, 'rb') as f:
        resp = pickle.load(f)

    ss.visrow(resp)
