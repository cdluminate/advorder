'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.

Black-box Order Attack Implementation
'''
try:
    from . import datasets
except ImportError as e:
    import datasets
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

USE_CPP_KERNEL = (int(os.getenv('USE_CPP_KERNEL', 0)) > 0)
USE_RUST_KERNEL = (int(os.getenv('USE_RUST_KERNEL', 0)) > 0)

'''
Select SRC implementation
In terms of speed: python < c++ < rust
priority         : python < c++ < rust
'''
if USE_CPP_KERNEL:
    try:
        from .srckernel_cc import *
    except ImportError:
        from srckernel_cc import *
if USE_RUST_KERNEL:
    try:
        from .srckernel_rs import *
    except ImportError:
        from srckernel_rs import *
if not USE_CPP_KERNEL and not USE_RUST_KERNEL:
    # USE_PY_KERNEL
    cprint('''\
Hint: `export USE_RUST_KERNEL=1` or `export USE_CPP_KERNEL=1` can
significantly speed up the black-box attack experiments.
That said, using C++/Rust kernel is optional for the experiments.''',
    'white', 'on_red', ['bold'])
    try:
        from .srckernel_py import *
    except ImportError:
        from srckernel_py import *


class BlackBoxRankingModel(object):
    '''
    input: query
    output: the foremost part of the sorted candidate list (partial)
    '''
    def __init__(self, candidates, canseek=-1, *, isadataset=False):
        raise NotImplementedError
    def __call__(self, query) -> th.Tensor:
        raise NotImplementedError


def RandSearch(model, query, rperm, *, eps=8./255., maxprobe=1e3,
        parallel:int=1, verbose=False) -> (th.Tensor, th.Tensor, float, float, tuple):
    '''
    Naive Random Search (Uniform distribution) for Reorder Attack.
    The other attacks should use the same function signature.
    There is no tunable hyper-parameters for this algorithm.

    Parameters

      model:callable        a BlackBoxRankingModel instance
      query:th.Tensor       the query image
      model.xcs:th.Tensor   immutable embedding of the candidates
      rperm:th.Tensor       the desired ranking of the attacker

    Output

      th.Tensor: perturbed query
      th.Tensor: perturbation itself
      float: score (sum of the Lower triangular)
      float: mean rank of the chosen candidates
      tuple: auxiliary information
    '''
    assert(parallel > 0)
    sys.stdout.flush()
    sys.stderr.flush()
    qr, xcs, canseek = query.clone().detach(), model.xcs, model.canseek
    model.model.eval()
    argsort, dist = model(qr)
    argrank = th.zeros_like(argsort, device='cpu')
    otopk = argsort[:len(rperm)]
    score0 = NearsightRankCorr(argsort, otopk, rperm)
    score = score0
    aux = (score, )
    if bool(os.getenv('NO_DR', 0)):
        dimreduce = False
        if query.nelement() > 1000:
            print('note, it is recommended to search in a low-dim space instead.')
    else:
        dimreduce = True if query.nelement() > 1000 else False
    if parallel == 1:
        for iter in tqdm(range(int(maxprobe))):
            if not dimreduce:
                # MNIST-like (-1,1,28,28)
                tmp = query + (2*th.rand_like(qr)-1) * eps
            else:
                # ImageNet-like (-1,3,224,224)
                _tmp = (2*th.rand(1,3,32,32, device=qr.device)-1) * eps
                _tmp = th.nn.functional.interpolate(_tmp, scale_factor=[7,7])
                tmp = query + _tmp
            tmp = tmp.clamp(min=0., max=1.)
            # Query
            argsort, dist = model(tmp)
            if canseek < 0:
                argrank = argsort.flatten().argsort()
                scoretmp = NearsightRankCorr(argsort, otopk, rperm)
            else:
                argrank = th.zeros_like(argsort, device='cpu')
                scoretmp = NearsightRankCorr(argsort, otopk, rperm)
            # Better?
            if scoretmp > score:
                print(colored(f'probe {iter}', 'blue'),
                        colored('accept | score', 'white', 'on_green'),
                        f'{score:.3f} ->', colored(f'{scoretmp:.3f}', 'blue'))
                print(colored(f'  & argsort[:topk]', 'blue'), argsort[:len(rperm)].cpu())
                qr = tmp
                score = scoretmp
            if score > 0.99:
                break  # early stop
        # Mean rank?
        if canseek < 0:
            meanrank = argrank[otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
    else: # parallel > 1
        if int(maxprobe)//parallel > 1:
            iterable = tqdm(range(int(maxprobe)//parallel))
        else:
            iterable = range(int(maxprobe)//parallel)
        for iter in iterable:
            if not dimreduce:
                # MNIST-like
                tmp = query + eps * (2*th.rand(parallel, *(qr.shape[1:]), device=qr.device)-1.0)
            else:
                # Imagenet-like
                _tmp = eps * (2*th.rand(parallel,3,32,32, device=qr.device)-1)
                _tmp = th.nn.functional.interpolate(_tmp, scale_factor=[7,7])
                tmp = query + _tmp
            tmp = tmp.clamp(min=0., max=1.)
            argsort, dist = model(tmp)
            if canseek < 0:
                argrank = argsort.argsort()
                scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
            else:
                argrank = th.zeros_like(argsort, device='cpu')
                scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
            if scoretmp.max() > score:
                print(colored(f'probe {iter}', 'blue'),
                        colored('accept | score', 'white', 'on_green'),
                        f'{score:.3f} ->', colored(f'{scoretmp.max():.3f}', 'blue'))
                print(colored(f'  & argsort[:topk]', 'blue'), argsort[scoretmp.argmax()][:len(rperm)].cpu())
                qr = tmp[scoretmp.argmax()].view(1, qr.shape[1], qr.shape[2], qr.shape[3])
                score = scoretmp[scoretmp.argmax()]
        if canseek < 0:
            meanrank = argrank[scoretmp.argmax()][otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
    if (error := (qr-query).abs().max()) > (eps + 1e-3):
        raise Exception(f'Perturbation out of bound: {error} > {eps}')
    if (qrmin := qr.min()) < 0. or (qrmax := qr.max()) > 1.:
        raise Exception(f'Adversarial example out of bound: min {qrmin}, max {qrmax}')
    score = max(score0, score)
    return (qr, qr - query, score, meanrank, aux)


def PSO(model:callable, query:th.Tensor, rperm:th.Tensor, *,
        eps=8./255., maxprobe=1e3,
        parallel:int=1, verbose=False) -> (th.Tensor, th.Tensor, float, float, tuple):
    '''
    {Basic} Particle Swarm Optimization (Metaheuristics) for Reorder Attack
    Reference: https://en.wikipedia.org/wiki/Particle_swarm_optimization
    obj: objective, f:R^n->R
    There are four tunables for this algorithm.
    ...
    <<< PSO param search strategy: Greedy : Factor 1000: 2 2 2 5 5 5
    <<< ref: https://hal.archives-ouvertes.fr/file/index/docid/764996/filename/SPSO_descriptions.pdf
    >> see bin/psoparam
    '''
    model.model.eval()
    if bool(os.getenv('NO_DR', 0)):
        dimreduce = False
        if query.nelement() > 1000:
            print('note, it is recommended to search in a low-dim space instead.')
    else:
        dimreduce = True if query.nelement() > 1000 else False
    # [[[ Parameters of PSO algorithm
    Npop = int(os.getenv('PSO_NPOP', 40))
    omega = float(os.getenv('PSO_OMEGA', 1.10))
    phip = float(os.getenv('PSO_PHIP', 0.57))
    phig = float(os.getenv('PSO_PHIG', 0.44))
    # ]]]
    with th.no_grad():
        sys.stdout.flush(); sys.stderr.flush()
        qr, xcs, canseek = query.clone().detach(), model.xcs, model.canseek
        argsort, dist = model(qr)
        argrank = th.zeros_like(argsort, device='cpu')
        otopk = argsort[:len(rperm)]
        score0 = NearsightRankCorr(argsort, otopk, rperm)
        score = score0
        aux = (score,)
        # [[ Initlize: particles, partibest, partiscore, globest, partivelo ]]
        C,H,W = qr.shape[1], qr.shape[2], qr.shape[3]
        # --- init particle position
        particles = ((th.rand(Npop,C,H,W,device=qr.device)*2-1)*eps
                + qr.view(-1,C,H,W)).clamp(min=0.,max=1.)
        particles = th.min(qr.expand(Npop,C,H,W) + eps, th.max(qr.expand(Npop,C,H,W) - eps, particles))
        # --- init particle best known position
        partibest = particles
        argsort, dist = model(particles)
        if canseek < 0:
            argrank = argsort.argsort()
        else:
            argrank = th.zeros_like(argsort, device='cpu')
        scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
        # --- init particle best known position scores
        partiscore = scoretmp
        print(colored(f'PSO> initial probe', 'blue'),
                    f'{score:.3f} ->', colored(f'{scoretmp.max():.3f}', 'blue'))
        print(colored(f'  & argsort[:topk]', 'blue'), argsort[scoretmp.argmax()][:len(rperm)].cpu())
        # --- init global best known position
        globest = particles[scoretmp.argmax()].view(1, qr.shape[1], qr.shape[2], qr.shape[3]).clone()
        # --- init global best known score
        score = scoretmp[scoretmp.argmax()]
        if canseek < 0:
            meanrank = argrank[scoretmp.argmax()][otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
        # --- init particle velocity
        partivelo = (th.rand_like(particles, device=particles.device)*2-1)*2*eps
        # start iterations
        for iter in tqdm(range((int(maxprobe) // Npop) - 1)):
            if parallel == 1:
                for j in range(Npop):
                    # pick random numbers
                    if not dimreduce:
                        rp = th.rand_like(qr, device=qr.device)
                        rg = th.rand_like(qr, device=qr.device)
                    else:
                        rp = th.rand((qr.shape[0], 3, 32 ,32), device=qr.device)
                        rp = th.nn.functional.interpolate(rp, scale_factor=[7,7])
                        rg = th.rand((qr.shape[0], 3, 32 ,32), device=qr.device)
                        rg = th.nn.functional.interpolate(rg, scale_factor=[7,7])
                    # update particle velocity
                    partivelo[j] = omega * partivelo[j] + \
                            phip * rp * (partibest[j] - particles[j]) + \
                            phig * rg * (globest - particles[j])
                    # update particle position
                    particles[j] = (particles[j] + partivelo[j]).clamp(min=0.,max=1.)
                    particles = th.min(qr.expand(Npop,C,H,W) + eps, th.max(qr.expand(Npop,C,H,W) - eps, particles))
                    # conditionally update partibest
                    argsort, dist = model(particles[j].view(-1,C,H,W))
                    if canseek < 0:
                        argrank = argsort.flatten().argsort()
                    else:
                        argrank = th.zeros_like(argsort, device='cpu')
                    scoretmp = NearsightRankCorr(argsort, otopk, rperm)
                    if canseek < 0:
                        meanrank = argrank[otopk].cpu().float().mean().item()
                    else:
                        meanrank = -1.0
                    if scoretmp > partiscore[j]:
                        partiscore[j] = scoretmp
                        partibest[j] = particles[j].view(-1,C,H,W)
                        #print(f'PSO> iter {iter}: partibest_{j} update: {scoretmp}')
                        # conditionally update globest
                        if scoretmp > score:
                            print(colored('PSO>', 'blue'), f'iter {iter}:',
                                    colored('globest update', 'white', 'on_green'),
                                    f': {score} ->', colored(f'{scoretmp}', 'blue'))
                            score = scoretmp
                            globest = partibest[j].view(-1,C,H,W)
                            if score > 0.99:
                                break
            elif parallel == Npop:
                # https://hal.archives-ouvertes.fr/file/index/docid/764996/filename/SPSO_descriptions.pdf
                # The standard PSO is serial, and we can modify this metaheuristic
                # making it parallel to speed up the attack. however the above
                # docuemnt points out that asynchronized PSO (due to parallelization)
                # performs a bit worse than the synchronized PSO (serial version).

                # XXX: BUT this is really significantly faster than the serial version!

                # pick random numbers
                if not dimreduce:
                    rp = th.rand((Npop, *qr.shape[1:]), device=qr.device)
                    rg = th.rand((Npop, *qr.shape[1:]), device=qr.device)
                else:
                    rp = th.rand((Npop, 3, 32, 32), device=qr.device)
                    rp = th.nn.functional.interpolate(rp, scale_factor=[7,7])
                    rg = th.rand((Npop, 3, 32, 32), device=qr.device)
                    rg = th.nn.functional.interpolate(rg, scale_factor=[7,7])
                # update particle velocity
                partivelo = omega * partivelo + \
                        phip * rp * (partibest - particles) + \
                        phig * rg * (globest - particles)
                # update particle position
                particles = (particles + partivelo).clamp(min=0., max=1.)
                particles = th.min(qr.expand(Npop,C,H,W)+eps, th.max(qr.expand(Npop,C,H,W)-eps, particles))
                # conditionally update the best results
                argsort, dist = model(particles.view(-1,C,H,W))
                if canseek < 0:
                    argrank = argsort.argsort()
                    scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
                else:
                    argrank = th.zeros_like(argsort, device='cpu')
                    scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
                for j in range(Npop):
                    if scoretmp[j] > partiscore[j]:
                        partiscore[j] = scoretmp[j]
                        partibest[j] = particles[j].view(-1,C,H,W)
                        if scoretmp[j] > score:
                            print(colored('PSO>', 'blue'), f'iter {iter}:',
                                    colored('globest update', 'white', 'on_green'),
                                    f': {score} ->', colored(f'{scoretmp[j]}', 'blue'))
                            score = scoretmp[j]
                            globest = partibest[j].view(-1,C,H,W)
                            if score > 0.99:
                                break
            else:
                raise ValueError(f"Parallel PSO, option parallel should equal 1 or Npop={Npop}")
    if (error := (globest-query).abs().max()) > (eps + 1e-3):
        raise Exception(f'Perturbation out of bound: {error} > {eps}')
    if (qrmin := globest.min()) < 0. or (qrmax := globest.max()) > 1.:
        raise Exception(f'Adversarial example out of bound: min {qrmin}, max {qrmax}')
    # return
    score = max(score0, score)
    return (globest, globest - qr, score, meanrank, aux)


def SPSA(model, query, rperm, *, eps=8./255., maxprobe=1e3,
        parallel:int=1, verbose=False) -> (th.Tensor, th.Tensor, float, float, tuple):
    '''
    SPSA attack. Adam optimizer may slightly boost the optimizer performance.
    Ref: https://github.com/thu-ml/realsafe
    '''
    # init: parameters and configuration
    qr, xcs, canseek = query.clone().detach(), model.xcs, model.canseek
    model.model.eval()
    if bool(os.getenv('NO_DR', 0)):
        dimreduce = False
        if query.nelement() > 1000:
            print('note, it is recommended to search in a low-dim space instead.')
    else:
        dimreduce = True if query.nelement() > 1000 else False
    # init: first evaluation
    argsort, dist = model(qr)
    argrank = th.zeros_like(argsort, device='cpu')
    otopk = argsort[:len(rperm)]
    score0 = NearsightRankCorr(argsort, otopk, rperm)
    score = score0
    aux = (score,)
    # init: SPSA params
    Npop = int(os.getenv('SS_NPOP', 50))
    # NOTE: you can change Npop to 48 when attacking JD SnapShop
    #       for more efficient use of the query budget.
    assert(Npop % 2 == 0)  # It should be a multiple of 2.
    lr = float(os.getenv('SS_LR', 2.))/255.
    sigma = float(os.getenv('SS_SIGMA', 2.))/255.
    mom = float(os.getenv('SS_MOM', 0.0))  # SGDM (+momentum)
    adam = bool(os.getenv('SS_ADAM', 0)) # Adam Optimizer
    assert(not all([adam == True, mom > 0.0]))
    # SPSA
    gbest = qr.clone().detach()
    with th.no_grad():
        pgd = qr.clone().detach()
        if mom > 0.:
            # initialize the tensors for the sgdm optimizer
            pgrad = th.zeros_like(pgd, device=qr.device)
        elif adam:
            # Initialize the tensors for the adam optimizer
            # we don't need to set these parameters via environment variables
            adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-9
            adam_m = th.zeros_like(pgd, device=qr.device)
            adam_v = th.zeros_like(pgd, device=qr.device)
            # adam_t is exactly the iteration num
            #adam_mhat = th.zeros_like(pgd, device=qr.device)
            #adam_vhat = th.zeros_like(pgd, device=qr.device)
        else:
            # simply PGD without anything
            pass
        for iteration in tqdm(range(int(maxprobe) // Npop)):
            # generate population
            if not dimreduce:
                perts = th.sign(th.randn((Npop//2, *qr.shape[1:]), device=qr.device))
            else:
                _tmp = th.sign(th.randn((Npop//2,3,32,32), device=qr.device))
                perts = th.nn.functional.interpolate(_tmp, scale_factor=[7,7])
            perts = th.cat([perts, -perts], dim=0).clamp(min=-1.0, max=+1.0)
            qx = (pgd + sigma * perts).clamp(min=0., max=1.)
            qx = th.min(qr.expand(Npop,*qr.shape[1:]) + eps, qx)
            qx = th.max(qr.expand(Npop,*qr.shape[1:]) - eps, qx)
            # evalute the samples
            argsort, dist = model(qx)
            if canseek < 0:
                argrank = argsort.argsort()
            else:
                argrank = th.zeros_like(argsort, device='cpu')
            scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
            # estimate gradient
            grad = (th.from_numpy(scoretmp).to(qr.device).view(-1,1,1,1) * perts).mean(dim=0)/sigma
            if mom > 0.:
                pgrad = grad + mom * pgrad  # SGDM
            elif adam:
                adam_m = adam_beta1 * adam_m + (1 - adam_beta1) * grad
                adam_v = adam_beta2 * adam_v + (1 - adam_beta2) * grad * grad
                adam_mhat = adam_m / (1 - adam_beta1 ** iteration)
                adam_vhat = adam_v / (1 - adam_beta2 ** iteration)
                pgrad = adam_mhat / (adam_vhat.sqrt() + adam_eps)
            else:
                pgrad = grad.clone().detach() # SGD
            pgd += (lr * (th.sign(pgrad))).clamp(min=-eps, max=+eps)
            pgd = th.max(qr - eps, th.min(qr + eps, pgd.float())).detach()
            pgd = pgd.clamp(min=0., max=1.)
            # report
            if scoretmp.max() > score:
                tqdm.write(' '.join([colored('SPSA>', 'blue'), f'Probe {iteration:3d}',
                        colored('ACCEPT', 'white', 'on_green'), 'score',
                        colored(f'{score:.3f}', 'yellow'),
                        '->',
                        colored(f'{scoretmp.max():.3f}', 'blue', None, ['bold'])]))
                gbest = qx[scoretmp.argmax()].view(1, *qr.shape[1:]).clone().detach()
                score = scoretmp.max()
            if score > 0.99:
                break
        if canseek < 0:
            meanrank = argrank[scoretmp.argmax()][otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
        # evaluate the last PGD stage (the 1001-th step)
        pgdargsort, _ = model(pgd)
        pgdscore = NearsightRankCorr(pgdargsort, otopk, rperm)
        if pgdscore > score:
            gbest = pgd
            score = pgdscore
            print(colored('SPSA>', 'yellow'), f'FINAL PGD STATE',
                    colored('ACCEPT', 'white', 'on_green'),
                    colored(f'=-> {score:.3f}', 'blue'))
    if (error := (gbest - query).abs().max()) > (eps + 1e-3):
        raise Exception(f'Perturbation out of bound: {error} > {eps}')
    if (qrmin := gbest.min()) < 0. or (qrmax := gbest.max()) > 1.:
        raise Exception(f'Adversarial example out of bound: min {qrmin}, max {qrmax}')
    score = max(score0, score)
    return (gbest, gbest - query, score, meanrank, aux)


def NES(model, query, rperm, *, eps=8./255., maxprobe=1e3,
        parallel:int=1, verbose=False) -> (th.Tensor, th.Tensor, float, float, tuple):
    '''
    NES attack (works well)
    Ref: https://github.com/thu-ml/realsafe
    see bin/nesparam for parameter search

    A momentum weighted 0.9 will also help improve the performance.
    e.g. Taux = 0.436 in the (5 -1 4) setting
    '''
    # init: parameters and configuration
    qr, xcs, canseek = query.clone().detach(), model.xcs, model.canseek
    model.model.eval()
    if bool(os.getenv('NO_DR', 0)):
        dimreduce = False
        if query.nelement() > 1000:
            print('note, it is recommended to search in a low-dim space instead.')
    else:
        dimreduce = True if query.nelement() > 1000 else False
    # init: first evaluation
    argsort, dist = model(qr)
    argrank = th.zeros_like(argsort, device='cpu')
    otopk = argsort[:len(rperm)]
    score0 = NearsightRankCorr(argsort, otopk, rperm)
    score = score0
    aux = (score,)
    # init: NES params
    Npop = int(os.getenv('NES_NPOP', 50))
    lr = float(os.getenv('NES_XLR', 2.))/255.
    sigma = eps/float(os.getenv('NES_XSIGMA', 0.5))
    mom = float(os.getenv('NES_MOM', 0.))  # momentum
    adam = bool(os.getenv('NES_ADAM', 1))  # Adam Optimizer
    assert(not all([adam == True, mom > 0.0]))
    # NES
    gbest = qr.clone().detach()
    with th.no_grad():
        # start the NESGrad + PGD process
        pgd = qr.clone().detach()
        if mom > 0.:
            # initialize tenors for the sgdm optimizer
            pgrad = th.zeros_like(pgd, device=qr.device)
        elif adam:
            adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-9
            adam_m = th.zeros_like(pgd, device=qr.device)
            adam_v = th.zeros_like(pgd, device=qr.device)
        else:
            # naive PGD without momentum
            pass
        for iteration in tqdm(range(int(maxprobe) // Npop)):
            # generate population
            if not dimreduce:
                perts = sigma * th.randn((Npop//2, *qr.shape[1:]), device=qr.device)
            else:
                _tmp = sigma * th.randn((Npop//2,3,32,32), device=qr.device)
                perts = th.nn.functional.interpolate(_tmp, scale_factor=[7,7])
            perts = th.cat([perts, -perts], dim=0).clamp(min=-eps, max=+eps)
            qx = (pgd + perts).clamp(min=0., max=1.)
            qx = th.min(qr.expand(Npop,*qr.shape[1:]) + eps, qx)
            qx = th.max(qr.expand(Npop,*qr.shape[1:]) - eps, qx)
            # evalute the samples
            argsort, dist = model(qx)
            if canseek < 0:
                argrank = argsort.argsort()
            else:
                argrank = th.zeros_like(argsort, device='cpu')
            scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
            # estimate gradient
            grad = (th.from_numpy(scoretmp).to(qr.device).view(-1, 1, 1, 1) * perts).mean(dim=0) / sigma
            if mom > 0.:
                # SGDM
                pgrad = grad + mom * pgrad
            elif adam:
                # Adam
                adam_m = adam_beta1 * adam_m + (1 - adam_beta1) * grad
                adam_v = adam_beta2 * adam_v + (1 - adam_beta2) * grad * grad
                adam_mhat = adam_m / (1 - adam_beta1 ** iteration)
                adam_vhat = adam_v / (1 - adam_beta2 ** iteration)
                pgrad = adam_mhat / (adam_vhat.sqrt() + adam_eps)
            else:
                # Bare PGD
                pgrad = grad.clone().detach()
            pgd += (lr * (th.sign(pgrad))).clamp(min=-eps, max=+eps)
            pgd = th.max(qr - eps, th.min(qr + eps, pgd.float())).detach()
            pgd = pgd.clamp(min=0., max=1.)
            # report
            if scoretmp.max() > score:
                print(colored('NES>', 'blue'), f'Probe {iteration}',
                        colored('ACCEPT', 'white', 'on_green'), f'score {score:.3f} ->',
                        colored(f'{scoretmp.max():.3f}', 'blue'))
                gbest = qx[scoretmp.argmax()].view(1, *qr.shape[1:]).clone().detach()
                score = scoretmp.max()
            if score > 0.99:
                break
        if canseek < 0:
            meanrank = argrank[scoretmp.argmax()][otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
        # evaluate the last PGD stage (the 1001-th step)
        pgdargsort, _ = model(pgd)
        pgdscore = NearsightRankCorr(pgdargsort, otopk, rperm)
        if pgdscore > score:
            gbest = pgd
            score = pgdscore
            print(colored('NES>', 'yellow'), f'FINAL PGD STATE',
                    colored('ACCEPT', 'white', 'on_green'),
                    colored(f'=-> {score:.3f}', 'blue'))
    if (error := (gbest - query).abs().max()) > (eps + 1e-3):
        raise Exception(f'Perturbation out of bound: {error} > {eps}')
    if (qrmin := gbest.min()) < 0. or (qrmax := gbest.max()) > 1.:
        raise Exception(f'Adversarial example out of bound: min {qrmin}, max {qrmax}')
    score = max(score0, score)
    return (gbest, gbest - query, score, meanrank, aux)


def Batk(model, query, rperm, *, eps=8./255., maxprobe=1e3,
        parallel:int=1, verbose=False) -> (th.Tensor, th.Tensor, float, float, tuple):
    '''
    Beta-Attack
    thanks to Wolfram alpha
    nabla_a log beta = diff(log((x^(a-1) * (1-x)^(b-1))/beta(a,b)), a)
    '''
    # init: parameters and configuration
    qr, xcs, canseek = query.clone().detach(), model.xcs, model.canseek
    model.model.eval()
    if bool(os.getenv('NO_DR', 0)):
        dimreduce = False
        if query.nelement() > 1000:
            print('note, it is recommended to search in a low-dim space instead.')
    else:
        dimreduce = True if query.nelement() > 1000 else False
    # init: first evaluation
    argsort, dist = model(qr)
    argrank = th.zeros_like(argsort, device='cpu')
    otopk = argsort[:len(rperm)]
    score0 = NearsightRankCorr(argsort, otopk, rperm)
    score = score0
    aux = (score,)
    # Parameters for B-Attack (distribution of ADVERSARIAL PERTURBATION)
    Npop = int(os.getenv('BA_NPOP', 50))
    if query.nelement() > 1000:
        lr = float(os.getenv('BA_LR', 3.0))  # default 3.0 on fashion
    else:
        lr = float(os.getenv('BA_LR', 0.5))  # default 0.5 on sop
    # N-attack
    gbest = qr.clone().detach()
    BETA_MIN = 1e-7
    BETA_MAX = 1e+3
    with th.no_grad():
        # alpha=beta=1, starting from the uniform distribution
        if not dimreduce:
            alpha = th.ones_like(qr, device=qr.device).view(1, qr.nelement())
            beta = th.ones_like(qr, device=qr.device).view(1, qr.nelement())
        else:
            alpha = th.ones((qr.shape[0], 3, 32, 32), device=qr.device).view(1, -1)
            beta = th.ones((qr.shape[0], 3, 32, 32), device=qr.device).view(1, -1)
        for iteration in tqdm(range(int(maxprobe) // Npop)):
            # generate purturbed samples
            xA = np.nan_to_num(alpha.clone().detach().cpu().numpy()).clip(min=BETA_MIN,max=BETA_MAX).repeat(Npop, axis=0)
            #print('xA', xA.shape, xA.min(), xA.max())
            xB = np.nan_to_num(beta.clone().detach().cpu().numpy()).clip(min=BETA_MIN,max=BETA_MAX).repeat(Npop, axis=0)
            #print('xB', xB.shape, xB.min(), xB.max())
            samples = th.from_numpy(np.random.beta(xA, xB)).to(qr.device) # (0,1)
            if dimreduce:
                samples = th.nn.functional.interpolate(samples.view(-1,3,32,32), scale_factor=[7,7])
            #print('sample max/min', samples.max(), samples.min())
            nsamples = samples * 2.0 - 1.0 # [-1,1]
            #print('sample max/min', samples.max(), samples.min())
            perts = (eps * nsamples).view(Npop, *qr.shape[1:]).to(qr.device)
            #print('perts max/min', perts.max(), perts.min())
            qx = (qr + perts).float().clamp(min=0., max=1.)
            # evalute the samples
            argsort, dist = model(qx)
            if canseek < 0:
                argrank = argsort.argsort()
            else:
                argrank = th.zeros_like(argsort, device='cpu')
            scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
            # update
            zscore = th.from_numpy(scoretmp).to(qr.device).view(Npop,1)
            if not dimreduce:
                batchgrada = th.digamma(alpha + beta).view(1,-1) - th.digamma(alpha).view(1,-1) \
                        + th.log(samples).view(Npop,-1)  # TODO: why nsamples better?
            else:
                batchgrada = th.digamma(alpha + beta).view(1,-1).repeat_interleave(7*7) - th.digamma(alpha).view(1,-1).repeat_interleave(7*7) \
                        + th.log(samples).view(Npop,-1)  # TODO: why nsamples better?
            grada = (zscore * batchgrada).clamp(min=-5,max=+5).mean(dim=0)
            #print('grada', grada.max(), grada.min())
            if not dimreduce:
                batchgradb = th.digamma(alpha + beta).view(1,-1) - th.digamma(beta).view(1,-1) \
                        + th.log(1-samples).view(Npop,-1) # TODO: why nsamples better?
            else:
                batchgradb = th.digamma(alpha + beta).view(1,-1).repeat_interleave(7*7) - th.digamma(beta).view(1,-1).repeat_interleave(7*7) \
                        + th.log(1-samples).view(Npop,-1) # TODO: why nsamples better?

            gradb = (zscore * batchgradb).clamp(min=-5,max=+5).mean(dim=0)
            #print(colored('Grad A>', 'yellow'), grada.mean().item(), grada.max().item(), grada.min().item())
            #print(colored('Grad B>', 'yellow'), gradb.mean().item(), gradb.max().item(), gradb.min().item())
            if not dimreduce:
                alpha = (alpha + lr * grada).clamp(min=BETA_MIN,max=BETA_MAX).clone()
                beta  = (beta + lr * gradb).clamp(min=BETA_MIN,max=BETA_MAX).clone()
            else:
                alpha = (alpha + lr * th.nn.functional.avg_pool1d(grada.view(1,1,-1),49,49,0).view(-1)).clamp(min=BETA_MIN,max=BETA_MAX).clone()
                beta = (beta + lr * th.nn.functional.avg_pool1d(gradb.view(1,1,-1),49,49,0).view(-1)).clamp(min=BETA_MIN,max=BETA_MAX).clone()
            # report
            if scoretmp.max() > score:
                print(colored('Beta>', 'blue'), f'Probe {iteration}',
                        colored('ACCEPT', 'white', 'on_green'), f'score {score:.3f} ->',
                        colored(f'{scoretmp.max():.3f}', 'blue'))
                gbest = qx[scoretmp.argmax()].view(1, *qr.shape[1:]).clone().detach()
                score = scoretmp.max()
            if score > 0.99:
                break
        if canseek < 0:
            meanrank = argrank[scoretmp.argmax()][otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
    if (error := (gbest - query).abs().max()) > (eps + 1e-3):
        raise Exception(f'Perturbation out of bound: {error} > {eps}')
    if (qrmin := gbest.min()) < 0. or (qrmax := gbest.max()) > 1.:
        raise Exception(f'Adversarial example out of bound: min {qrmin}, max {qrmax}')
    score = max(score0, score)
    return (gbest, gbest - query, score, meanrank, aux)


def Uatk(model, query, rperm, *, eps=8./255., maxprobe=1e3,
        parallel:int=1, verbose=False) -> (th.Tensor, th.Tensor, float, float, tuple):
    '''
    Uniform-Attack (deprecated)
    '''
    # init: parameters and configuration
    qr, xcs, canseek = query.clone().detach(), model.xcs, model.canseek
    model.model.eval()
    dimreduce = True if query.nelement() > 1000 else False
    if dimreduce:
        raise NotImplementedError
    # init: first evaluation
    argsort, dist = model(qr)
    argrank = th.zeros_like(argsort, device='cpu')
    otopk = argsort[:len(rperm)]
    score0 = NearsightRankCorr(argsort, otopk, rperm)
    score = score0
    aux = (score,)
    # Parameters for N-Attack (distribution of ADVERSARIAL EXAMPLE)
    Npop = int(os.getenv('NA_NPOP', 100))
    lr = float(os.getenv('NA_LR', 0.01 * 1e8))
    # N-attack
    with th.no_grad():
        alpha = th.zeros_like(qr, device=qr.device)
        beta = th.ones_like(qr, device=qr.device)
        for iteration in tqdm(range(int(maxprobe) // Npop)):
            # generate purturbed samples
            perts = eps * ((alpha + (beta-alpha) * th.rand((Npop, *qr.shape[1:]), device=qr.device))*2. - 1.0)
            qx = (query + perts).clamp(min=0., max=1.)
            # evalute the samples
            argsort, dist = model(qx)
            if canseek < 0:
                argrank = argsort.argsort()
            else:
                raise NotImplementedError
            scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
            # update
            zscore = th.from_numpy(scoretmp).to(qr.device).view(Npop,1,1,1)
            grada = (zscore / (beta - alpha + 1e7)).mean(dim=0)
            gradb = (-zscore / (beta - alpha + 1e7)).mean(dim=0)
            #print(colored('Grad A>', 'yellow'), grada.mean().item(), grada.max().item(), grada.min().item())
            #print(colored('Grad B>', 'yellow'), gradb.mean().item(), gradb.max().item(), gradb.min().item())
            alpha += lr * grada
            beta  += lr * gradb
            alpha, beta = th.min(alpha, beta), th.max(alpha, beta)
            alpha, beta = alpha.clamp(min=0.,max=1.), beta.clamp(min=0.,max=1.)
            #print(alpha.flatten()[0], beta.flatten()[0])
            # report
            if scoretmp.max() > score:
                print(colored('Natk>', 'blue'), f'Probe {iteration}',
                        colored('ACCEPT', 'white', 'on_green'), f'score {score:.3f} ->',
                        colored(f'{scoretmp.max():.3f}', 'blue'))
                qr = qx[scoretmp.argmax()].view(1, *qr.shape[1:]).clone().detach()
                score = scoretmp.max()
        if canseek < 0:
            meanrank = argrank[scoretmp.argmax()][otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
    if (error := (qr - query).abs().max()) > (eps + 1e-3):
        raise Exception(f'Perturbation out of bound: {error} > {eps}')
    if (qrmin := qr.min()) < 0. or (qrmax := qr.max()) > 1.:
        raise Exception(f'Adversarial example out of bound: min {qrmin}, max {qrmax}')
    score = max(score0, score)
    return (qr, qr - query, score, meanrank, aux)


def Natk(model, query, rperm, *, eps=8./255., maxprobe=1e3,
        parallel:int=1, verbose=False) -> (th.Tensor, th.Tensor, float, float, tuple):
    '''
    N-Attack for Reorder Attack (requires pytorch > 1.6.0 due to th.atanh) (deprecated)
    Ref: https://github.com/thu-ml/realsafe
    '''
    # init: parameters and configuration
    qr, xcs, canseek = query.clone().detach(), model.xcs, model.canseek
    model.model.eval()
    dimreduce = True if query.nelement() > 1000 else False
    if dimreduce:
        raise NotImplementedError
    # init: first evaluation
    argsort, dist = model(qr)
    argrank = th.zeros_like(argsort, device='cpu')
    otopk = argsort[:len(rperm)]
    score0 = NearsightRankCorr(argsort, otopk, rperm)
    score = score0
    aux = (score,)
    # Parameters for N-Attack (distribution of ADVERSARIAL EXAMPLE)
    Npop = int(os.getenv('NA_NPOP', 100))
    lr = float(os.getenv('NA_LR', 4./255.))
    sigma = float(os.getenv('NA_SIGMA', eps/10))
    # N-attack
    with th.no_grad():
        mu = th.atanh(qr.clone().detach()) # instead of th.randn_like(qr, device=qr.device)
        for iteration in tqdm(range(int(maxprobe) // Npop)):
            # generate purturbed samples
            aadvs = mu + th.randn((Npop, *qr.shape[1:]), device=qr.device) * sigma
            nadvs = (th.tanh(aadvs) + 1.) / 2.
            deltas = (nadvs - query).clamp(min=-eps, max=+eps)
            qx = (query + deltas).clamp(min=0., max=1.)
            # evalute the samples
            argsort, dist = model(qx)
            if canseek < 0:
                argrank = argsort.argsort()
            else:
                raise NotImplementedError
            scoretmp = BatchNearsightRankCorr(argsort, otopk, rperm)
            # update
            zscore = th.from_numpy((scoretmp - scoretmp.mean())/(1e+7+scoretmp.std())).to(qr.device)
            grad = (zscore.view(Npop,1,1,1) * (aadvs - mu)).mean(dim=0)
            mu += lr * grad
            # report
            if scoretmp.max() > score:
                print(colored('Natk>', 'blue'), f'Probe {iteration}',
                        colored('ACCEPT', 'white', 'on_green'), f'score {score:.3f} ->',
                        colored(f'{scoretmp.max():.3f}', 'blue'))
                qr = qx[scoretmp.argmax()].view(1, *qr.shape[1:]).clone().detach()
                score = scoretmp.max()
        if canseek < 0:
            meanrank = argrank[scoretmp.argmax()][otopk].cpu().float().mean().item()
        else:
            meanrank = -1.0
    if (error := (qr - query).abs().max()) > (eps + 1e-3):
        raise Exception(f'Perturbation out of bound: {error} > {eps}')
    if (qrmin := qr.min()) < 0. or (qrmax := qr.max()) > 1.:
        raise Exception(f'Adversarial example out of bound: min {qrmin}, max {qrmax}')
    score = max(score0, score)
    return (qr, qr - query, score, meanrank, aux)

####################### Real BlackBox Model ###################################

class JDSnapShop(BlackBoxRankingModel):
    '''
    https://aidoc.jd.com/image/snapshop.html
    This is a special class
    '''
    def __init__(self, candidates, canseek=-1, *, isadataset=False):
        from .snapshop import JDQuery as jdquery
        self.model = jdquery
        assert(candidates is None)
        assert(isadataset == False)
        assert(canseek <= 50 and canseek > 0)  # we only support 50
        self.canseek = canseek
    def __call__(self, query) -> th.Tensor:
        im = query.detach().cpu().numpy()
        res = self.model(im)
        sims = res.json()['result']['dataValue'][0]['sims']
        argsort = th.LongTensor([int(can['skuId']) for can in sims])
        return argsort[:self.canseek], th.zeros(self.canseek)
    @staticmethod
    def getloader(kind, batchsize):
        return iter([])


class Fashion(BlackBoxRankingModel):
    def __init__(self, candidates, canseek=-1, *, isadataset=False, device='cpu'):
        from . import faC_c2f2
        self.device = device
        self.model = faC_c2f2.Model().to(device)
        self.model.load_state_dict(th.load(f'trained/faC_c2f2.sdth'))
        self.xcs = []
        if isadataset:
            print('| Initializing the candidate embeddings ...', end=' ')
            with th.no_grad():
                for (images, labels) in tqdm(candidates):
                    embs = self.model.forward(images.to(device))
                    embs = th.nn.functional.normalize(embs, dim=1, p=2)
                    self.xcs.append(embs)
                self.xcs = th.cat(self.xcs)
            print(f'got {len(self.xcs)} emb vectors')
        else:
            self.xcs = candidates  # should be the pre-calculated embeddings
        self.canseek = canseek
    def __call__(self, query) -> th.Tensor:
        with th.no_grad():
            if query.shape[0] == 1:
                xq = self.model(query.to(self.device))
                scores = th.nn.functional.cosine_similarity(xq, self.xcs).flatten()
                argsort = scores.argsort(dim=0, descending=True)
                if self.canseek < 0:
                    return argsort, scores[argsort]
                else:
                    return argsort[:self.canseek], scores[argsort][:self.canseek]
            else:
                xq = self.model(query.to(self.device))
                nxq = th.nn.functional.normalize(xq, p=2, dim=1)
                nxcs = th.nn.functional.normalize(self.xcs, p=2, dim=1)
                scores = th.mm(nxq, nxcs.t())
                argsort = scores.argsort(descending=True)
                if self.canseek < 0:
                    return argsort, th.cat([x[argsort[i]] for (i,x) in enumerate(scores)])
                else:
                    return argsort[:, :self.canseek], th.cat([x[argsort[i]][:self.canseek]
                        for (i,x) in enumerate(scores)])
    @staticmethod
    def getloader(kind:str='test', batchsize:int=100):
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'test':
            return datasets.fashion.get_loader(
                    os.path.expanduser(config['fashion-mnist']['path']),
                    batchsize, 't10k')
        else:
            raise NotImplementedError


class FashionLenet(BlackBoxRankingModel):
    def __init__(self, candidates, canseek=-1, *, isadataset=False, device='cpu'):
        from . import faC_lenet
        self.device = device
        self.model = faC_lenet.Model().to(device)
        self.model.load_state_dict(th.load(f'trained/faC_lenet.sdth'))
        self.xcs = []
        if isadataset:
            print('| Initializing the candidate embeddings ...', end=' ')
            with th.no_grad():
                for (images, labels) in tqdm(candidates):
                    embs = self.model.forward(images.to(device))
                    embs = th.nn.functional.normalize(embs, dim=1, p=2)
                    self.xcs.append(embs)
                self.xcs = th.cat(self.xcs)
            print(f'got {len(self.xcs)} emb vectors')
        else:
            self.xcs = candidates  # should be the pre-calculated embeddings
        self.canseek = canseek


class FashionResnet(BlackBoxRankingModel):
    def __init__(self, candidates, canseek=-1, *, isadataset=False, device='cpu'):
        from . import faC_res18
        self.device = device
        self.model = faC_res18.Model().to(device)
        self.model.load_state_dict(th.load(f'trained/faC_res18.sdth'))
        self.model.eval()
        self.xcs = []
        if isadataset:
            print('| Initializing the candidate embeddings ...', end=' ')
            with th.no_grad():
                for (images, labels) in tqdm(candidates):
                    embs = self.model.forward(images.to(device))
                    embs = th.nn.functional.normalize(embs, dim=1, p=2)
                    self.xcs.append(embs)
                self.xcs = th.cat(self.xcs)
            print(f'got {len(self.xcs)} emb vectors')
        else:
            self.xcs = candidates  # should be the pre-calculated embeddings
        self.canseek = canseek


class FashionE(BlackBoxRankingModel):
    def __init__(self, candidates, canseek=-1, *, isadataset=False, device='cpu'):
        from . import faE_c2f2
        self.device = device
        self.model = faE_c2f2.Model().to(device)
        self.model.load_state_dict(th.load(f'trained/faE_c2f2.sdth'))
        self.xcs = []
        if isadataset:
            print('| Initializing the candidate embeddings ...', end=' ')
            with th.no_grad():
                for (images, labels) in tqdm(candidates):
                    embs = self.model.forward(images.to(device))
                    self.xcs.append(embs)
                self.xcs = th.cat(self.xcs)
            print(f'got {len(self.xcs)} emb vectors')
        else:
            self.xcs = candidates  # should be the pre-calculated embeddings
        self.canseek = canseek
    def __call__(self, query) -> th.Tensor:
        with th.no_grad():
            if query.shape[0] == 1:
                xq = self.model(query.to(self.device))
                scores = th.nn.functional.pairwise_distance(xq, self.xcs, p=2).flatten()
                argsort = scores.argsort(dim=0, descending=True)
                if self.canseek < 0:
                    return argsort, scores[argsort]
                else:
                    return argsort[:self.canseek], scores[argsort][:self.canseek]
            else:
                N, D, NX = query.size(0), self.xcs.size(1), self.xcs.size(0)
                xq = self.model(query.to(self.device))
                xq = xq.view(N, 1, D).expand(N, NX, D)
                xcs = self.xcs.view(1, NX, D).expand(N, NX, D)
                scores = (xq - xcs).norm(2, dim=2)
                argsort = scores.argsort(descending=True)
                if self.canseek < 0:
                    return argsort, th.cat([x[argsort[i]] for (i,x) in enumerate(scores)])
                else:
                    return argsort[:, :self.canseek], th.cat([x[argsort[i]][:self.canseek]
                        for (i,x) in enumerate(scores)])
    @staticmethod
    def getloader(kind:str='test', batchsize:int=100):
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'test':
            return datasets.fashion.get_loader(
                    os.path.expanduser(config['fashion-mnist']['path']),
                    batchsize, 't10k')
        else:
            raise NotImplementedError


class Sop(BlackBoxRankingModel):
    def __init__(self, candidates, canseek=-1, *, isadataset=False, device='cpu'):
        from . import sopE_res18
        self.device = device
        self.model = sopE_res18.Model().to(device)
        self.model.load_state_dict(th.load(f'trained/sopE_res18.sdth'))
        self.model.eval()
        self.xcs = []
        if isadataset:
            print('! Initializing the candidate embeddings ...')
            if os.path.exists(f'trained/sopE_res18.xcs.cache'):
                self.xcs = th.load(f'trained/sopE_res18.xcs.cache')
                print(f'! Got {len(self.xcs)} emb vectors from the CACHE file.')
            else:
                with th.no_grad():
                    for (images, labels) in tqdm(candidates):
                        embs = self.model.forward(images.to(device))
                        self.xcs.append(embs)
                self.xcs = th.cat(self.xcs)
                print(f'! Got {len(self.xcs)} emb vectors, oven-fresh.')
                th.save(self.xcs, f'trained/sopE_res18.xcs.cache')
        else:
            self.xcs = candidates # should be pre-calculated embeddings
        self.canseek = canseek
    def __call__(self, query) -> th.Tensor:
        canseek = self.canseek
        with th.no_grad():
            if query.shape[0] == 1:
                xq = self.model(query.to(self.device))
                scores = th.nn.functional.pairwise_distance(xq, self.xcs, p=2).flatten()
                argsort = scores.argsort(dim=0, descending=False)
                if self.canseek < 0:
                    return argsort, scores[argsort]
                else:
                    return argsort[:canseek], scores[argsort][:canseek]
            else:
                N, D, NX = query.size(0), self.xcs.size(1), self.xcs.size(0)
                xq = self.model(query.to(self.device))
                # [ High memory consumption
                #xq = xq.view(N,1,D).expand(N,NX,D)
                #xcs = self.xcs.view(1,NX,D).expand(N,NX,D)
                #scores = (xq - xcs).norm(2, dim=2)
                # ]
                # [ Lower memory consumption divide and conquer
                xq1 = xq[:N//2].view(N//2,1,D).expand(N//2,NX,D)
                xcs1 = self.xcs.view(1,NX,D).expand(N//2,NX,D)
                scores1 = (xq1 - xcs1).norm(2, dim=2)
                #print(scores1.shape)
                xq2 = xq[N//2:].view(N-N//2,1,D).expand(N-N//2,NX,D)
                xcs2 = self.xcs.view(1,NX,D).expand(N-N//2,NX,D)
                scores2 = (xq2 - xcs2).norm(2, dim=2)
                #print(scores2.shape)
                scores = th.cat([scores1, scores2])
                #print(scores.shape)
                argsort = scores.argsort(descending=False)
                if self.canseek < 0:
                    return argsort, th.cat([x[argsort[i]] for (i, x) in enumerate(scores)])
                else:
                    return argsort[:, :self.canseek], th.cat([x[argsort[i]][:self.canseek]
                        for (i,x) in enumerate(scores)])
    @staticmethod
    def getloader(kind:str='test', batchsize:int=100):
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'test':
            return datasets.sop.get_loader(
                    os.path.expanduser(config['sop']['path']),
                    batchsize, 1, 'test')
        else:
            raise NotImplementedError


######################### TESTS ###############################################


class LinearModel(BlackBoxRankingModel):
    def __init__(self, candidates, canseek=-1, *, isadataset=False):
        self.model = th.nn.Linear(32, 16)  # embedding size 16
        self.candidates = candidates
        if isadataset:
            raise NotImplementedError
        else:
            with th.no_grad():
                self.xcs = self.model(self.candidates)
        self.canseek = canseek
    def __call__(self, query) -> th.Tensor:
        with th.no_grad():
            xq = self.model(query)
            scores = th.nn.functional.cosine_similarity(xq, self.xcs).flatten()
            argsort = scores.argsort(dim=0, descending=True)
        if self.canseek < 0:
            # can see the whole ranking list
            return argsort, scores[argsort]
        else:
            return argsort[:self.canseek], th.cat([x[argsort[i]][:self.canseek] for i,x in enumerate(scores)])


def test_algos(M=5):
    '''
    let's construct a simple ranking model and let our algorithm attack it.
    '''
    cprint(f'[[ Sanity Test of Reorder Attack Algos | M = {M} ]]',
            'white', None, ['bold'])
    query = th.rand(1, 32)
    candidates = th.rand(50, 32)
    model = LinearModel(candidates, canseek=-1)

    argsort, dist = model(query)
    print(colored('top argsort', 'yellow'), argsort[:M+1], '...')

    rtopk = th.randperm(M)
    print(colored('      rperm', 'yellow'), rtopk)
    print(colored('     resort', 'yellow'), argsort[rtopk])

    for algo in [RandSearch, Batk, PSO, NES, SPSA]:
        cprint(f'--- {algo}', 'red', None, ['bold'])
        qr, r, score, mrank, _ = algo(model, query, rtopk,
                eps=1.0, maxprobe=1e3, verbose=True)


def benchmark_src(M=128):
    '''
    benchmark SRC implementation
    '''
    import time
    import rich
    c = rich.get_console()
    x = th.randint(1000, (50,))
    y = th.randint(1000, (50,))
    r = th.randperm(25)
    tm_start = time.time()
    s = NearsightRankCorr(x, y, r)
    tm_one = time.time() - tm_start
    c.print('  tm_one', tm_one)
    X = th.randint(1000, (50, 50))
    tm_start = time.time()
    S = BatchNearsightRankCorr(X, y, r)
    tm_batch = time.time() - tm_start
    c.print('tm_batch', tm_batch)


if __name__ == '__main__':
    benchmark_src()
    #import doctest
    #doctest.testmod(verbose=True)


