#!/usr/bin/env python3
'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import sys, os, yaml, re
import numpy as np
import torch as th, torch.utils.data
import argparse, collections
from tqdm import tqdm
import lib
from termcolor import cprint, colored


def Attack(argv):
    '''
    Attack a pre-trained model
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device', type=str,
            default='cuda' if th.cuda.is_available() else 'cpu')
    ag.add_argument('-A', '--attack', type=str, required=True,
            choices=[ # order attack (pure)
                'SPO:PGD-M5', 'SPO:PGD-M10', 'SPO:PGD-M25',
                ])
    ag.add_argument('-e', '--epsilon', default=4./255.,
            type=float, help='hyper-param epsilon | 1412.6572-FGSM min 0.007')
    ag.add_argument('-M', '--model', type=str, required=True)
    ag.add_argument('-T', '--transfer', type=str, required=False, default='')
    ag.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    ag.add_argument('--vv', action='store_true', help='more verbose')
    ag = ag.parse_args(argv)

    print('>>> Parsing arguments and configuration file')
    for x in yaml.dump(vars(ag)).split('\n'): cprint(x, 'green')
    if ag.vv: ag.verbose = True
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)
    cprint(f'Attacking method is {ag.attack} \u03b5={ag.epsilon}', 'white', 'on_magenta')

    # Load the white-box attacking target model
    if re.match('\S+:\S+', ag.model):
        Mname, Mpath = re.match('(\S+):(\S+)', ag.model).groups()
    else:
        Mname, Mpath = ag.model, 'trained/' + ag.model + '.sdth'
    print(f'>>> Loading white-box target {Mname} model from:', Mpath)
    model = getattr(lib, Mname).Model().to(ag.device)
    model.load_state_dict(th.load(Mpath))
    print(model)

    if ag.transfer:
        if re.match('\S+:\S+', ag.transfer):
            # also specified the path of the blackbox model
            Tname, Tpath = re.match('(\S+):(\S+)', ag.transfer).groups()
        else:
            # load it from the default path
            Tname, Tpath = ag.transfer, 'trained/'+ag.transfer+'.sdth'
        print(f'>>> Loading {Tname} from {Tpath} for blackbox/transfer attack.')
        modelT = getattr(lib, Tname).Model().to(ag.device)
        modelT.load_state_dict(th.load(Tpath))
        modelT.eval()

    print('>>> Loading dataset ...', end=' ')
    if not ag.vv:
        loader_test = \
            model.getloader('test', config[Mname]['batchsize_atk'])
        if ag.attack in ('Q:PGD-M1000', 'Q:PGD-M10000', 'Q:PGD-MX'):
            loader_test = model.getloader('test', 10) # override batchsize
    elif ag.vv:
        loader_test = model.getloader('test', 1)
        print('| overriden batchsize to 1', end=' ')
    print('| Testing dataset size =', len(loader_test.dataset))

    print('>>> Start Attacking ...')
    dconf = {'epsilon': ag.epsilon}
    if ag.transfer:
        dconf['TRANSFER'] = {'transfer': Tname, 'model': modelT, 'device': ag.device}
    model.attack(ag.attack, loader_test, dconf=dconf, verbose=ag.verbose)


if __name__ == '__main__':
    Attack(sys.argv[1:])
