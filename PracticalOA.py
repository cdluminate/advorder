#!/usr/bin/env python3
'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import sys, os, yaml, re, json
import numpy as np, torch as th
from lib import reorder
import lib
import argparse, collections
from termcolor import cprint, colored
import statistics
from PIL import Image
from torchvision.transforms import functional as transfunc
import rich
c = rich.get_console()


def PracticalAttack(argv):
    '''
    Attack a pre-trained model
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-A', '--attack', type=str, default='SPSA')
    ag.add_argument('-e', '--epsilon', default=1./255., type=float)
    '''
    A NOTE ON SELECTION OF EPSILON (For Attacking JDModel)

    0.062 (16/255) -> top5 go out of sight with little exception
    0.031 ( 8/255) -> 1-of-top5 does not go out of sight
    0.015 ( 4/255) -> 3-of-top5 does not go out of sight
    0.008 ( 2/255) -> top5 within sight but not close to each other
    0.004 ( 1/255) -> quite good. (and cannot be lower)

    For BingModel
    1/255  -> topk very persistent
    2/255  -> top3 very persistent
    4/255  -> top1 starts to vary
    8/255  -> looks appropriate.
    '''
    ag.add_argument('-M', '--model', type=str, choices=['JDModel', 'BingModel'])
    ag.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    ag.add_argument('-Q', '--qbudget', type=int, default=500, help='query budget')
    ag.add_argument('-k', '--topk', type=int, default=5, help='generate permutation for topk')
    ag.add_argument('-c', '--canseek', type=int, default=50, help='length of returned ranking list')
    ag.add_argument('-l', '--payload', type=str, required=True, help='path to the payload image')
    ag.add_argument('-V', '--visualize', action='store_true')
    ag.add_argument('-O', '--oneshot', action='store_true')
    ag.add_argument('--randperm', action='store_true', help='use a random permutation instead')
    ag = ag.parse_args(argv)
    cprint(json.dumps(vars(ag), indent=4), 'yellow')

    # Process the arguments
    if ag.epsilon > 1.0:
        ag.epsilon = ag.epsilon / 255.
    assert(ag.topk > 1)

    # Load the payload image
    image = Image.open(ag.payload, mode='r').resize((224,224), Image.ANTIALIAS)
    query = transfunc.to_tensor(image).clone().unsqueeze(0)
    print(f'* Payload Image Info: shape={query.shape}')
    #tmp = transfunc.to_pil_image(query.squeeze(), mode='RGB')
    #tmp.show()
    #input('2')

    # Load the target model
    cprint(f'Setting up the "{ag.model}" Model')
    if ag.model == 'JDModel':
        model = getattr(lib.snapshop, ag.model)(canseek=ag.canseek)
    elif ag.model == 'BingModel':
        model = getattr(lib.bing, ag.model)(canseek=ag.canseek)
    else:
        raise ValueError('unsupported model')
    print(model)

    # Start attacking
    cprint(f'>_< Starting {ag.attack} Attack with Epsilon = {ag.epsilon:.3f}',
            'red', None, ['bold', 'underline'])
    argsort, _ = model(query, id='init')
    orig_argsort = argsort.clone().detach()
    if not ag.randperm:
        rperm = th.LongTensor([1, 5, 4, 3, 2]) - 1  # manually specified order
    else:
        rperm = np.arange(ag.topk)
        np.random.shuffle(rperm)
        rperm = th.from_numpy(rperm)
    otopk = argsort[:len(rperm)]
    rtopk = otopk[rperm]
    cprint(f'> Original CanSee\n {argsort.tolist()}', 'cyan')
    cprint(f'> Original  TopK {otopk}', 'green')
    cprint(f'> Attacker Rperm {rperm}', 'yellow')
    cprint(f'> Expected  TopK {rtopk}', 'red')
    if ag.oneshot:
        print('Exiting as requested oneshot mode.')
        exit(0)

    qr, r, score, mrank, aux = getattr(reorder, ag.attack)(model, query, rperm,
            eps=ag.epsilon, parallel=1, maxprobe=ag.qbudget, verbose=True)
    #argsort, _ = model(query, id='final')
    #cprint(f'> FINAL TopK', 'red')
    #cprint(argsort.tolist(), 'cyan')
    c.print('Final score:', score)


if __name__ == '__main__':
    PracticalAttack(sys.argv[1:])
