#!/usr/bin/env python3
'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import sys, os, yaml
import numpy as np
import torch as th, torch.utils.data
import argparse, collections
from tqdm import tqdm
import lib
from termcolor import cprint, colored


def Train(argv):
    '''
    Train the Neural Network
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device',
            default='cuda' if th.cuda.is_available() else 'cpu',
            type=str, help='computational device')
    ag.add_argument('-M', '--model', type=str, required=True)
    ag.add_argument('-A', '--attack', type=float, default=None, required=False)
    ag.add_argument('--overfit', action='store_true')
    ag.add_argument('--report', type=int, default=10)
    ag.add_argument('--validate', action='store_true')
    ag = ag.parse_args(argv)

    print('>>> Parsing arguments')
    for x in yaml.dump(vars(ag)).split('\n'): cprint(x, 'green')
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)

    if ag.validate:
        sdpath = 'trained/' + ag.model + '.sdth'
        print('>>> Loading model from', sdpath)
        model = getattr(lib, ag.model).Model().to(ag.device)
        model.load_state_dict(th.load(sdpath))
        print(model)
        print('>>> Loading datasets')
        loader_test = model.getloader('test', config[ag.model]['batchsize'])
        print(len(loader_test.dataset))
        cprint(f'Validate {model.validate(loader_test)}', 'white', 'on_magenta')
        exit(0)

    print('>>> Setting up model and optimizer')
    model = getattr(lib, ag.model).Model().to(ag.device)
    optim = th.optim.Adam(model.parameters(),
            lr=config[ag.model]['lr'], weight_decay=1e-7)
    print(model); print(optim)

    print('>>> Loading datasets')
    loader_train = model.getloader('train', config[ag.model]['batchsize'])
    loader_test  = model.getloader('test', config[ag.model]['batchsize'])
    print(len(loader_train.dataset), len(loader_test.dataset))

    print('>>> Start training')
    cprint(f'Validate[-1] {model.validate(loader_test)}', 'white', 'on_magenta')
    if not os.path.exists('trained'):
        os.mkdir('trained')
    for epoch in range(config[ag.model]['epoch']):

        # dynamic learning rate
        dylr = int(config[ag.model].get('dylr', -1))
        if dylr > 0:
            lrn = config[ag.model]['lr']
            lrn = lrn * 0.1 if epoch >= dylr else lrn
            for param_group in optim.param_groups:
                param_group['lr'] = lrn

        # Do the normal training process
        for iteration, (images, labels) in enumerate(loader_train):
            model.train()
            if ag.attack is None:
                output, loss = model.loss(images, labels)
            else:
                output, loss = model.loss_adversary(images, labels, eps=ag.attack)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (iteration % ag.report == 0) or ag.overfit:
                model.report(epoch, iteration, len(loader_train),
                        output, labels, loss)
            if ag.overfit:
                break

        # save a snapshot
        cprint(f'Validate[{epoch}] '+str(model.validate(loader_test)), 'white', 'on_magenta')
        th.save(model.state_dict(), 'trained/'+ag.model+'+snapshot.sdth')

    print('>>> Saving the network to:', 'trained/' + ag.model + '.sdth')
    th.save(model.cpu().state_dict(), 'trained/' + ag.model + '.sdth')


if __name__ == '__main__':
    Train(sys.argv[1:])
