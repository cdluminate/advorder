'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import sys
import random
import glob
import argparse
import rich
c = rich.get_console()

if __name__ == '__main__':
    ag = argparse.ArgumentParser()
    ag.add_argument('-p', '--pool', type=str,
            default=os.path.expanduser('~/.torch/Stanford_Online_Products'))
    ag.add_argument('-s', '--suffix', type=str, default='.JPG')
    ag = ag.parse_args(sys.argv[1:])
    #c.print(ag)

    with c.status('Globbing files ...'):
        p = os.path.join(ag.pool, '**')
        files = glob.glob(p, recursive=True)
        files = list(filter(lambda x: x.endswith(ag.suffix), files))
    #c.print(f'Found {len(files)} {ag.suffix} files.')
    c.print(random.choice(files))
