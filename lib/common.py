'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os, sys, re
import functools
import torch as th
import collections
from tqdm import tqdm
import pylab as lab
import traceback
import math
import statistics
from scipy import stats
import numpy as np
import random
from .utils import IMmean, IMstd, renorm, denorm, xdnorm, chw2hwc
from termcolor import cprint, colored


def rank_attack(model, attack, loader, *, dconf, device, verbose=False):
    '''
    generic attack method for embedding/ranking models
    '''
    # >> pre-process the options
    normimg = dconf.get('normimg', False)
    if dconf.get('metric', None) is None:
        raise ValueError('dconf parameter misses the "metric" key')
    candidates = model.compute_embedding(loader, device=device,
            l2norm=(True if dconf['metric']=='C' else False))
    if dconf.get('TRANSFER', None) is None:
        dconf['TRANSFER'] = None
    else:
        candidates_trans = dconf['TRANSFER']['model'].compute_embedding(
                loader, device=dconf['TRANSFER']['device'],
                l2norm=(True if 'C' in dconf['TRANSFER']['transfer'] else False))
        dconf['TRANSFER']['candidates'] = candidates_trans
    ruthless = int(os.getenv('RUTHLESS', -1))  # maxiter for attack

    # >> dispatch: attacking
    print('>>> Candidate Set:', candidates[0].shape, candidates[1].shape)
    correct_orig, correct_adv, total = 0, 0, 0
    rankup, embshift, prankgt, prank_trans = [], [], [], []
    for N, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
        #if N < random.randint(0, 60502//2): continue # picking sample for vis
        #if N < 14676//2: continue
        if (ruthless > 0) and (N >= ruthless):
            break
        if verbose: cprint('\n'+'\u2500'*64, 'cyan')
        if re.match('^Q.?:PGD-M\d+$', attack) is not None:
            regroup = re.match('^Q(.?):PGD-M(\d+)$', attack).groups()
            pm = str(regroup[0]) # + / -
            assert(pm in ['+', '-'])
            M = int(regroup[1]) # m, num of candidates
            assert(M > 0)
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='QA', M=M, pm=pm,
                    transfer=dconf['TRANSFER'])
        elif re.match('^SPQ.?:PGD-M\d+$', attack) is not None:
            regroup = re.match('^SPQ(.?):PGD-M(\d+)$', attack).groups()
            pm = str(regroup[0]) # + / -
            assert(pm in ['+', '-'])
            M = int(regroup[1]) # m, num of candidates
            assert(M > 0)
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='SPQA', M=M, pm=pm,
                    transfer=dconf['TRANSFER'])
        elif re.match('^F:PGD-M\d+$', attack) is not None:
            regroup = re.match('^F:PGD-M(\d+)$', attack).groups() 
            M = int(regroup[0]) # m, num of candidates
            assert(M > 1)
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='FOA', M=M, pm=None,
                    transfer=dconf['TRANSFER'])
        elif re.match('^SPO:PGD-M\d+$', attack) is not None:
            regroup = re.match('^SPO:PGD-M(\d+)$', attack).groups()
            M = int(regroup[0]) # m, num of candidates
            xr, r, out, loss, count = RankPGD(model, images, labels,
                    candidates, eps=dconf['epsilon'], verbose=verbose,
                    device=device, loader=loader, metric=dconf['metric'],
                    normimg=normimg, atype='SPFOA', M=M, pm=None,
                    transfer=dconf['TRANSFER'])
        else:
            raise ValueError(f"Attack {attack} unsupported.")
        correct_orig += count[0][0]
        correct_adv += count[1][0]
        total += len(labels)
        rankup.append(count[1][1])
        embshift.append(count[1][2])
        prankgt.append(count[1][3])
        prank_trans.append(count[1][4])
        if N*images.shape[0] > 10000: break  # XXX: N=10000 for speed
    total = max(1,total)
    # >> report overall attacking result on the test dataset
    cprint('\u2500'*64, 'cyan')
    if int(os.getenv('IAP', 0)) > 0:
        cprint(' '.join([f'Summary[{attack} \u03B5={dconf["epsilon"]}]:',
                'white-box=', '%.3f'%statistics.mean(rankup), # abuse var
                'black-box=', '%.3f'%statistics.mean(embshift), # abuse var
                'white-box-orig=', '%.3f'%statistics.mean(prankgt), # abuse var
                'black-box-orig=', '%.3f'%statistics.mean(prank_trans), # abuse var
                ]), 'cyan')
    else:
        cprint(' '.join([f'Summary[{attack} \u03B5={dconf["epsilon"]}]:',
                'baseline=', '%.3f'%(100.*(correct_orig/total)),
                'adv=', '%.3f'%(100.*(correct_adv/total)),
                'advReduce=', '%.3f'%(100.*(correct_orig - correct_adv) / total),
                'rankUp=', '%.3f'%statistics.mean(rankup),
                'embShift=', '%.3f'%statistics.mean(embshift),
                'prankgt=', '%.3f'%statistics.mean(prankgt),
                'prank_trans=', '%.3f'%statistics.mean(prank_trans),
                ]), 'cyan')
    cprint('\u2500'*64, 'cyan')


class LossFactory(object):
    '''
    Factory of loss functions used in all ranking attacks
    '''
    @staticmethod
    def RankLossEmbShift(repv: th.tensor, repv_orig: th.tensor, *, metric: str):
        '''
        Computes the embedding shift, we want to maximize it by gradient descent
        '''
        if metric == 'C':
            distance = 1 - th.mm(repv, repv_orig.t())
            loss = -distance.trace() # gradient ascent on trace, i.e. diag.sum
        elif metric == 'E':
            distance = th.nn.functional.pairwise_distance(repv, repv_orig, p=2)
            loss = -distance.sum()
        return loss
    @staticmethod
    def RankLossQueryAttack(qs: th.tensor, Cs: th.tensor, Xs: th.tensor, *, metric: str, pm: str,
            dist: th.tensor = None, cidx: th.tensor = None):
        '''
        Computes the loss function for pure query attack
        '''
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1])
        NIter, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        DO_RANK = (dist is not None) and (cidx is not None)
        losses, ranks = [], []
        #refrank = []
        for i in range(NIter):
            #== compute the pairwise loss
            q = qs[i].view(1, D)  # [1, output_1]
            C = Cs[i, :, :].view(M, D)  # [1, output_1]
            if metric == 'C':
                A = (1 - th.mm(q, C.t())).expand(NX, M)
                B = (1 - th.mm(Xs, q.t())).expand(NX, M)
            elif metric == 'E':
                A = (C - q).norm(2, dim=1).expand(NX, M)
                B = (Xs - q).norm(2, dim=1).view(NX, 1).expand(NX, M)
            #== loss function
            if '+' == pm:
                loss = (A-B).clamp(min=0.).mean()
            elif '-' == pm:
                loss = (-A+B).clamp(min=0.).mean()
            losses.append(loss)
            #== compute the rank
            if DO_RANK:
                ranks.append(th.mean(dist[i].flatten().argsort().argsort()
                    [cidx[i,:].flatten()].float()).item())
            #refrank.append( ((A>B).float().mean()).item() )
        #print('(debug)', 'rank=', statistics.mean(refrank))
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks) if DO_RANK else None
        return loss, rank
    @staticmethod
    def RankLossFullOrderM2Attack(qs: th.tensor, ps: th.tensor, ns: th.tensor, *, metric: str):
        '''
        Computes the loss function for M=2 full-order attack
        '''
        assert(qs.shape[0] == ps.shape[0] == ns.shape[0])
        assert(qs.shape[1] == ps.shape[1] == ns.shape[1])
        Batch, D = qs.shape[0], qs.shape[1]
        if metric == 'C':
            dist1 = 1 - th.nn.functional.cosine_similarity(qs, ps, dim=1)
            dist2 = 1 - th.nn.functional.cosine_similarity(qs, ns, dim=1)
        elif metric == 'E':
            dist1 = th.nn.functional.pairwise_distance(qs, ps, p=2)
            dist2 = th.nn.functional.pairwise_distance(qs, ns, p=2)
        else:
            raise ValueError(metric)
        loss = (dist1 - dist2).clamp(min=0.).mean()
        acc = (dist1 <= dist2).sum().item() / Batch
        return loss, acc
    @staticmethod
    def RankLossFullOrderMXAttack(qs: th.tensor, Cs: th.tensor, *, metric=str):
        assert(qs.shape[1] == Cs.shape[2])
        NIter, M, D = qs.shape[0], Cs.shape[1], Cs.shape[2]
        losses, taus = [], []
        for i in range(NIter):
            q = qs[i].view(1, D)
            C = Cs[i, :, :].view(M, D)
            if metric == 'C':
                dist = 1 - th.mm(q, C.t())
            elif metric == 'E':
                dist = (C - q).norm(2, dim=1)
            tau = stats.kendalltau(np.arange(M), dist.cpu().detach().numpy())[0]
            taus.append(tau)
            dist = dist.expand(M, M)
            loss = (dist.t() - dist).triu(diagonal=1).clamp(min=0.).mean()
            losses.append(loss)
        loss = th.stack(losses).mean()
        tau = statistics.mean(x for x in taus if not math.isnan(x))
        return loss, tau
    def __init__(self, request: str):
        '''
        Initialize various loss functions
        '''
        self.funcmap = {
                'QA': self.RankLossQueryAttack,
                'QA+': functools.partial(self.RankLossQueryAttack, pm='+'),
                'QA-': functools.partial(self.RankLossQueryAttack, pm='-'),
                'FOA2': self.RankLossFullOrderM2Attack,
                'FOAX': self.RankLossFullOrderMXAttack,
                }
        if request not in self.funcmap.keys():
            raise KeyError(f'Requested loss function "{request}" not found!')
        self.request = request
    def __call__(self, *args, **kwargs):
        return self.funcmap[self.request](*args, **kwargs)


## MARK: STAGE0
def RankPGD(model, images, labels, candi, *,
        eps=0.3, alpha=1./255., atype=None, M=None, W=None, pm=None,
        verbose=False, device='cpu', loader=None, metric=None,
        normimg=False, transfer=None):
    '''
    Perform FGSM/PGD Query/Candidate attack on the given batch of images, L_infty constraint
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/fast_gradient_method.py

    This is the core of the adversarial ranking implementation,
    but we don't have enough energy to tidy it up before ICCV submission.
    '''
    # >> prepare the current batch of images
    assert(type(images) == th.Tensor)
    images = images.clone().detach().to(device)
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)
    # >> sanity check for normalized images, if any
    if normimg:
        # normed_img = (image - mean)/std
        IMmean = th.tensor([0.485, 0.456, 0.406], device=device)
        IMstd = th.tensor([0.229, 0.224, 0.225], device=device)
        renorm = lambda im: im.sub(IMmean[:,None,None]).div(IMstd[:,None,None])
        denorm = lambda im: im.mul(IMstd[:,None,None]).add(IMmean[:,None,None])
    if (not normimg) and ((images > 1.0).sum() + (images < 0.0).sum() > 0):
        raise Exception("please toggle 'normimg' as True for sanity")
    def tensorStat(t):
        return f'Min {t.min().item()} Max {t.max().item()} Mean {t.mean().item()}'

    #<<<<<< STAGE1: ORIG SAMPLE EVALUATION <<<<<<
    model.eval()
    with th.no_grad():

        # -- [orig] -- forward the original samples with the original loss
        # >> Result[output]: embedding vectors
        # >> Result[dist]: distance matrix (current batch x database)
        if metric == 'C':
            output = model.forward(images, l2norm=True)
            dist = 1 - output @ candi[0].t()  # [num_output_num, num_candidate]
        elif metric == 'E':
            output = model.forward(images, l2norm=False)
            dist = []
            # the memory requirement is insane if we want to do the pairwise distance
            # matrix in a single step like faC_c2f2_siamese.py's loss function.
            for i in range(output.shape[0]):
                xq = output[i].view(1, -1)
                xqd = (candi[0] - xq).norm(2, dim=1).squeeze()
                dist.append(xqd)
            dist = th.stack(dist)  # [num_output_num, num_candidate]
        else:
            raise ValueError(metric)
        output_orig = output.clone().detach()
        dist_orig = dist.clone().detach()
        loss = th.tensor(-1) # we don't track this value anymore
        loss_orig = th.tensor(-1) # we don't track this value anymore

        #== <transfer> forward the samples with the transfer model
        if transfer is not None:
            if 'C' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=True)
                dist_trans = 1 - output_trans @ transfer['candidates'][0].t()
            elif 'E' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=False)
                dist_trans = []
                for i in range(output_trans.shape[0]):
                    xtrans = output_trans[i].view(1, -1)
                    xdtrans = (transfer['candidates'][0] - xtrans).norm(2, dim=1).squeeze()
                    dist_trans.append(xdtrans)
                dist_trans = th.stack(dist_trans)

        # -- [orig] -- select attack targets and calculate the attacking loss
        if (atype in ['FOA', 'SPFOA']) and (M is not None) and (M == 2):
            # -- [orig] ranking attack, M=2

            #== configuration for FOA:M=2 
            M_GT = 5 # sensible choice due to SOP dataset property
            XI = float(os.getenv('SP', 10.)) # balancing the "SP" and "QA" component
            if 'SP' not in atype:
                XI = None  # override SP weight to None

            #== select the M=2 candidates. note, x1 is closer to q than x2
            if True:
                # local sampling (default)
                topmost = int(candi[0].size(0) * 0.01)
                topxm = dist.topk(topmost+1, dim=1, largest=False)[1][:,1:]  # [output_0, M]
                sel = np.vstack([np.random.permutation(topmost) for j in range(topxm.shape[0])])
                msample = th.stack([topxm[i][np.sort(sel[i,:M])] for i in range(topxm.shape[0])])
                if 'SP' in atype:
                    mgtruth = th.stack([topxm[i][np.sort(sel[i,M:])[:M_GT]] for i in range(topxm.shape[0])])
            else:
                # global sampling
                distsort = dist.sort(dim=1)[1]  # [output_0, candi_0]
                mpairs = th.randint(candi[0].shape[0], (output.shape[0], M)).sort(dim=1)[0] # [output_0, M]
                msample= th.stack([distsort[i, mpairs[i]] for i in range(output.shape[0])])  # [output_0, M]
                if 'SP' in atype:
                    mgtruth = dist.topk(M_GT+1, dim=1, largest=False)[1][:,1:]  # [output_0, M_GT]
            embpairs = candi[0][msample, :] # [output_0, M, output_1]
            if 'SP' in atype:
                embgts = candi[0][mgtruth, :]  # [output_0, M_GT, output_1]

            # >> compute the (ordinary) loss on selected targets
            loss, acc = LossFactory('FOA2')(output, embpairs[:,1,:], embpairs[:,0,:], metric=metric)
            #== Semantic preserving? (SP)
            if 'SP' in atype:
                loss_sp, rank_gt = LossFactory('QA+')(output, embgts, candi[0],
                        metric=metric, dist=dist, cidx=mgtruth)
                loss = loss + XI * loss_sp
                prankgt_orig = rank_gt #/ candi[0].size(0)

            # >> backup and report
            correct_orig = acc * output.shape[0]
            loss_orig = loss.clone().detach()
            if verbose:
                print()
                if 'SP' not in atype:
                    print('* Original Sample', 'loss=', loss.item(), 'FOA:Accu=', acc)
                else:
                    print('* Original Sample', 'loss=', loss.item(), 'where loss_sp=', loss_sp.item(),
                            'FOA:Accu=', acc, 'GT.R@mean=', rank_gt)

            # <transfer>
            if transfer is not None:
                embpairs_trans = transfer['candidates'][0][msample, :]
                _, acc_trans = LossFactory('FOA2')(output_trans, embpairs_trans[:,1,:], embpairs_trans[:,0,:],
                        metric=('C' if 'C' in transfer['transfer'] else 'E'))
                if 'SP' not in atype:
                    print('* <transfer> Original Sample', 'FOA:Accu=', acc_trans)
                else:
                    embgts_trans = transfer['candidates'][0][mgtruth, :]
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm='+',
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                    print('* <transfer> Original Sample', 'FOA:Accu=', acc_trans,
                            'GT.R@mean=', rank_sp_trans)

        elif (atype in ['FOA', 'SPFOA']) and (M is not None) and (M > 2):
            # -- [orig] ranking attack, M>2

            #== configuration for FOA:M>2
            M_GT = 5 # sensible choice due to SOP dataset property
            XI = float(os.getenv('SP', 10.)) # balancing the "SP" and "QA" component
            if 'SP' not in atype:
                XI = None  # override SP weight to None

            #== select M>2 candidates, in any order
            if True:
                # Just select the original top-k
                topxm = dist.topk(M, dim=1, largest=False)[1]
                rpm = np.stack([np.random.permutation(M) for j in range(topxm.shape[0])])
                msample = th.stack([topxm[i][rpm[i]] for i in range(topxm.shape[0])])
                if 'SP' in atype:
                    mgtruth = msample
            elif False:
                # local sampling (from the topmost 1% samples)
                topmost = int(candi[0].size(0) * 0.01)
                topxm = dist.topk(topmost+1, dim=1, largest=False)[1][:,1:]  # [output_0, M]
                sel = np.vstack([np.random.permutation(topmost) for j in range(topxm.shape[0])])
                msample = th.stack([topxm[i][sel[i,:M]] for i in range(topxm.shape[0])])
                if 'SP' in atype:
                    mgtruth = th.stack([topxm[i][np.sort(sel[i,M:])[:M_GT]] for i in range(topxm.shape[0])])
            else:
                # global sampling
                msample = th.randint(candi[0].shape[0], (output.shape[0], M)) # [output_0, M]
                if 'SP' in atype:
                    mgtruth = dist.topk(M_GT+1, dim=1, largest=False)[1][:,1:]
            embpairs = candi[0][msample, :] # [output_0, M, output_1]
            if 'SP' in atype:
                embgts = candi[0][mgtruth, :] # [output_0, M_GT, output_1]

            # >> adversarial inequalities formed by every pair of samples
            loss, tau = LossFactory('FOAX')(output, embpairs, metric=metric)
            #== Semantic preserving? (SP)
            if 'SP' in atype:
                loss_sp, rank_sp = LossFactory('QA+')(output, embgts, candi[0],
                        metric=metric, dist=dist, cidx=mgtruth)
                loss = loss + XI * loss_sp
                prankgt_orig = rank_sp #/ candi[0].size(0)

            # >> backup and report
            correct_orig = tau * output.shape[0] / 100.
            loss_orig = loss.clone().detach()
            if verbose:
                print()
                if 'SP' not in atype:
                    print('* Original Sample', 'loss=', loss.item(), 'FOA:tau=', tau)
                else:
                    print('* Original Sample', 'loss=', loss.item(), 'where loss_sp=', loss_sp.item(),
                            'FOA:tau=', tau, 'GT.R@mean=', rank_sp)
            # <transfer>
            if transfer is not None:
                embpairs_trans = transfer['candidates'][0][msample, :]
                _, tau_trans = LossFactory('FOAX')(output_trans, embpairs_trans,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'))
                if 'SP' not in atype:
                    print('* <transfer> Original Sample', 'FOA:tau=', tau_trans)
                else:
                    embgts_trans = transfer['candidates'][0][mgtruth, :]
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm='+',
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                    print('* <transfer> Original Sample', 'FOA:tau=', tau_trans,
                            'GT.R@mean=', rank_sp_trans)

        elif (atype in ['QA', 'SPQA']) and (M is not None):
            #== semantic-preserving (SP) query attack
            #; the pure query attack has a downside: its semantic may be changed
            #; during the attack. That would result in a very weird ranking result,
            #; even if the attacking goal was achieved.

            #== configuration
            M_GT = 5 # sensible due to the SOP dataset property
            XI = float(os.getenv('SP', 1. if ('+' == pm) else 100.)) # balancing the "SP" and "QA" loss functions
            if 'SP' not in atype:
                XI = None

            #== first, select the attacking targets and the ground-truth vectors for
            #; the SP purpose.
            if '+' == pm:
                # random sampling from populationfor QA+
                if 'global' == os.getenv('SAMPLE', 'global'):
                    msample = th.randint(candi[0].shape[0], (output.shape[0], M)) # [output_0,M]
                elif 'local' == os.getenv('SAMPLE', 'global'):
                    local_lb = int(candi[0].shape[0]*0.01)
                    local_ub = int(candi[0].shape[0]*0.05)
                    topxm = dist.topk(local_ub+1, dim=1, largest=False)[1][:,1:]
                    sel = np.random.randint(local_lb, local_ub, (output.shape[0], M))
                    msample = th.stack([topxm[i][sel[i]] for i in range(topxm.shape[0])])
                if 'SP' in atype:
                    mgtruth = dist.topk(M_GT+1, dim=1, largest=False)[1][:,1:] # [output_0, M]
            elif '-' == pm:
                # random sampling from top-3M for QA-
                topmost = int(candi[0].size(0) * 0.01)
                if int(os.getenv('VIS', 0)) > 0:
                    topmost = int(candi[0].size(0) * 0.0003)
                if transfer is None:
                    topxm = dist.topk(topmost+1, dim=1, largest=False)[1][:,1:]
                else:
                    topxm = dist_trans.topk(topmost+1, dim=1, largest=False)[1][:,1:]
                sel = np.vstack([np.random.permutation(topmost) for i in range(output.shape[0])])
                msample = th.stack([topxm[i][sel[i,:M]] for i in range(topxm.shape[0])])
                if 'SP' in atype:
                    mgtruth = th.stack([topxm[i][np.sort(sel[i,M:])[:M_GT]] for i in range(topxm.shape[0])])
            embpairs = candi[0][msample, :]
            if 'SP' in atype:
                embgts = candi[0][mgtruth, :]

            #== evaluate the SPQA loss on original samples
            if 'SP' in atype:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss_sp, rank_sp = LossFactory('QA+')(output, embgts, candi[0],
                        metric=metric, dist=dist, cidx=mgtruth)
                loss = loss_qa + XI * loss_sp
            else:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss = loss_qa

            #== overall loss function of the batch
            mrank = rank_qa / candi[0].shape[0]
            correct_orig = mrank * output.shape[0] / 100.
            loss_orig = loss.clone().detach()
            if 'SP' in atype:
                mrankgt = rank_sp / candi[0].shape[0]
                sp_orig = mrankgt * output.shape[0] / 100.
                prankgt_orig = mrankgt
            #== backup and report
            if verbose:
                print()
                if 'SP' in atype:
                    print('* Original Sample', 'loss=', loss.item(),
                            f'SPQA{pm}:rank=', mrank,
                            f'SPQA{pm}:GTrank=', mrankgt)
                else:
                    print('* Original Sample', 'loss=', loss.item(),
                            f'QA{pm}:rank=', mrank)

            # <transfer>
            if transfer is not None:
                embpairs_trans = transfer['candidates'][0][msample, :]
                _, rank_qa_trans = LossFactory('QA')(output_trans, embpairs_trans,
                        transfer['candidates'][0], pm=pm,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'),
                        dist=dist_trans, cidx=msample)
                if 'SP' in atype:
                    embgts_trans = transfer['candidates'][0][mgtruth, :]
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm=pm,
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                if 'SP' not in atype:
                    print('* <transfer> Original Sample', f'QA{pm}:rank=', rank_qa_trans / candi[0].shape[0])
                else:
                    print('* <transfer> Original Sample', f'SPQA{pm}:rank=', rank_qa_trans / candi[0].shape[0],
                            f'SPQA{pm}:GTrank=', rank_sp_trans / candi[0].shape[0])
        else:
            raise Exception("Unknown attack")
    # >>>>>>>>>>>>>>>>>>>>>>>>>> ORIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # <<<<<<<<<<<<<<<<<<<<<<<<<< MARK: STATTACK <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # -- [attack] start attack, maxiter is the number of PGD iterations (FGSM: maxiter=1)

    # -- [alpha/epsilon] PGD parameter tricks balancing yield v.s. time consumption
    #    Note, 1/255 \approx 0.004 
    if True:
        alpha = 1./255.
        maxiter = 24
    elif eps > 1./255.:
        alpha = float(min(max(eps/10., 1./255.), 0.01))  # PGD hyper-parameter
        maxiter = int(min(max(10, 2*eps/alpha), 30))  # PGD hyper-parameter
    elif eps < 1./255.:
        maxiter = 1
        alpha = eps / max(1, maxiter//2)  # doesn't attack

    # this ensures the best attacking result, but is very time-consuming
    if int(os.getenv('PGD_UTMOST', 0)) > 0:
        print('| Overriding PGD Parameter for the utmost attacking result|')
        alpha = 1./255.
        maxiter = int(os.getenv('PGD_UTMOST'))

    # PGD with/without random init?
    if int(os.getenv('RINIT', 0))>0:
        if not normimg:
            images = images + eps*2*(0.5-th.rand(images.shape)).to(images.device)
            images = th.clamp(images, min=0., max=1.)
            images = images.detach()
            images.requires_grad = True
        else:
            images = images + (eps/IMstd[:,None,None])*2*(0.5-th.rand(images.shape)).to(images.device)
            images = th.max(images, renorm(th.zeros(images.shape).to(device)))
            images = th.min(images, renorm(th.ones(images.shape).to(device)))
            images = images.detach()
            images.requires_grad = True

    # truning the model into tranining mode triggers obscure problem:
    # incorrect validation performance due to BatchNorm. We do automatic
    # differentiation in evaluation mode instead.
    #model.train()
    for iteration in range(maxiter):

        # >> prepare optimizer for SGD
        optim = th.optim.SGD(model.parameters(), lr=1.)
        optimx = th.optim.SGD([images], lr=1.)
        optim.zero_grad(); optimx.zero_grad()
        output = model.forward(images, l2norm=(True if metric=='C' else False))

        if (atype in ['FOA', 'SPFOA']) and (M is not None) and (M == 2): # -- [attack] query attack, M=2
            # >> reverse the inequalities (ordinary: d1 < d2, adversary: d1 > d2)
            loss, _ = LossFactory('FOA2')(output, embpairs[:,1,:], embpairs[:,0,:], metric=metric)
            if 'SP' in atype:
                loss_sp, _ = LossFactory('QA')(output, embgts, candi[0], metric=metric, pm='+')
                loss = loss + XI * loss_sp
            if verbose:
                if 'SP' in atype:
                    cprint(' '.join(['(PGD) iter', str(iteration), 'loss=', str(loss.item()),
                        '\t|optim|', 'loss:SP(QA+)=', str(loss_sp.item())]), 'grey')
                else:
                    cprint(' '.join(['(PGD) iter', str(iteration), 'loss', str(loss.item())]), 'grey')
        elif (atype in ['FOA', 'SPFOA']) and (M is not None) and (M > 2): # -- [attack] query attack, M>2
            # >> enforce the random inequality set (di < dj for all i,j where i<j)
            loss, _ = LossFactory('FOAX')(output, embpairs, metric=metric)
            if 'SP' in atype:
                loss_sp, _ = LossFactory('QA')(output, embgts, candi[0], metric=metric, pm='+')
                loss = loss + XI * loss_sp
            if verbose:
                if 'SP' in atype:
                    cprint(' '.join(['(PGD) iter %3d'%iteration, 'loss %.10f'%loss.item(),
                        '\t|optim|', 'loss:SP(QA+)= %.10f'%loss_sp.item()]), 'cyan')
                else:
                    cprint(' '.join(['(PGD) iter', str(iteration), 'loss', str(loss.item())]), 'cyan')
        elif (atype in ['QA', 'SPQA']) and (M is not None):
            #== enforce the target set of inequalities, while preserving the semantic
            if int(os.getenv('DISTANCE', 0)) > 0:
                if 'SP' in atype:
                    #raise NotImplementedError("SP for distance based objective makes no sense here")
                    loss_qa, _ = LossFactory('QA-DIST')(output, embpairs, candi[0], metric=metric, pm=pm)
                    loss_sp, _ = LossFactory('QA-DIST')(output, embgts, candi[0], metric=metric, pm='+')
                    loss = loss_qa + XI * loss_sp
                    if verbose:
                        cprint(['(PGD) iter', str(iteration), 'loss', str(loss.item())].join(' '), 'grey')
                else:
                    loss, _ = LossFactory('QA-DIST')(output, embpairs, candi[0], metric=metric, pm=pm)
                    if verbose:
                        cprint(['(PGD) iter', str(iteration), 'loss', str(loss.item())].join(' '), 'grey')
            elif 'SP' in atype:
                loss_qa, _ = LossFactory('QA')(output, embpairs, candi[0], metric=metric, pm=pm)
                loss_sp, _ = LossFactory('QA')(output, embgts, candi[0], metric=metric, pm='+')
                loss = loss_qa + XI * loss_sp
                if verbose:
                    cprint(['(PGD) iter', str(iteration), 'loss', str(loss.item()),
                        '\t|optim|', f'loss:QA{pm}=', str(loss_qa.item()),
                        'loss:SP(QA+)=', str(loss_sp.item())].join(' '), 'grey')
            else:
                loss, _ = LossFactory('QA')(output, embpairs, candi[0], metric=metric, pm=pm)
                if verbose:
                    cprint(['(PGD) iter', str(iteration), 'loss', str(loss.item())].join(' '), 'grey')
        else:
            raise Exception("Unknown attack")
        # >> Do FGSM/PGD update step
        # >> PGD projection when image has been normalized is a little bit special
        loss.backward()
        if not normimg:
            if maxiter>1:
                images.grad.data.copy_(alpha * th.sign(images.grad))
            elif maxiter==1:
                images.grad.data.copy_(eps * th.sign(images.grad))  # FGSM
            # >> PGD: project SGD optimized result back to a valid region
            optimx.step()
            images = th.min(images, images_orig + eps) # L_infty constraint
            images = th.max(images, images_orig - eps) # L_infty constraint
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
        else: #normimg:
            if maxiter>1:
                images.grad.data.copy_((alpha/IMstd[:,None,None])*th.sign(images.grad))
            elif maxiter==1:
                images.grad.data.copy_((eps/IMstd[:,None,None])*th.sign(images.grad))
            else:
                raise Exception("Your arguments must be wrong")
            # >> PGD: project SGD optimized result back to a valid region
            optimx.step()
            images = th.min(images, images_orig + (eps/IMstd[:,None,None]))
            images = th.max(images, images_orig - (eps/IMstd[:,None,None]))
            #images = th.clamp(images, min=0., max=1.)
            images = th.max(images, renorm(th.zeros(images.shape).to(device)))
            images = th.min(images, renorm(th.ones(images.shape).to(device)))
            images = images.clone().detach()
            images.requires_grad = True

    # >>>>>>>>>>>>>>>>>>>>>>>> END ATTACK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # <<<<<<<<<<<<<<<<<<<<<<<< ADVERSARIAL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # --- [eval adv] re-evaluate these off-manifold adversarial samples
    xr = images
    r = images - images_orig
    model.eval()
    with th.no_grad():
        # >> forward the adversarial examples, and optionally calculate loss
        if metric == 'C':
            output = model.forward(images, l2norm=True)
            dist = 1 - output @ candi[0].t()
            loss = th.tensor(-1)
        elif metric == 'E':
            output = model.forward(images, l2norm=False)
            dist = []
            # the memory requirement is insane if we want to do the pairwise distance
            # matrix in a single step like faC_c2f2_siamese.py's loss function.
            for i in range(output.shape[0]):
                xq = output[i].view(1, -1)
                xqd = (candi[0] - xq).norm(dim=1).squeeze()
                dist.append(xqd)
            dist = th.stack(dist)
            loss = th.tensor(-1)
        else:
            raise ValueError(metric)
        output_adv = output.clone().detach()
        dist_adv = dist.clone().detach()

        # <transfer>
        if transfer is not None:
            if 'C' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=True)
                dist_trans = 1 - output_trans @ transfer['candidates'][0].t()
            elif 'E' in transfer['transfer']:
                output_trans = transfer['model'].forward(images, l2norm=False)
                dist_trans = []
                for i in range(output_trans.shape[0]):
                    xtrans = output_trans[i].view(1, -1)
                    xdtrans = (transfer['candidates'][0] - xtrans).norm(2, dim=1).squeeze()
                    dist_trans.append(xdtrans)
                dist_trans = th.stack(dist_trans)

        # >> calculate embedding shift
        if metric == 'C':
            distance = 1 - th.mm(output, output_orig.t())
            embshift = distance.trace()/output.shape[0] # i.e. trace = diag.sum
            embshift = embshift.item()
        elif metric == 'E':
            distance = th.nn.functional.pairwise_distance(output, output_orig, p=2)
            embshift = distance.sum()/output.shape[0]
            embshift = embshift.item()

        if (atype in ['FOA', 'SPFOA']) and (M is not None) and (M == 2):
            # -- [eval adv] ranking attack, M=2
            loss, acc = LossFactory('FOA2')(output, embpairs[:,1,:], embpairs[:,0,:], metric=metric)
            #== Semantic preserving? (SP)
            if 'SP' in atype:
                loss_sp, rank_sp = LossFactory('QA')(output, embgts, candi[0],
                        metric=metric, pm='+', dist=dist, cidx=mgtruth)
                loss = loss + XI * loss_sp
                prankgt_adv = rank_sp #/ candi[0].size(0)

            correct_adv = acc * output.shape[0]
            loss_adv = loss.clone().detach()
            rankup = 0
            if verbose:
                if 'SP' not in atype:
                    print('* Adversarial Example', 'loss=', loss.item(),
                            'FOA:Accuracy', acc, 'embShift=', embshift)
                else:
                    print('* Adversarial Example', 'loss=', loss.item(),
                            'where loss_sp=', loss_sp.item(),
                            'FOA:Accuracy', acc, 'GT.R@mean=', rank_sp)
            # <transfer>
            if transfer is not None:
                _, acc_trans = LossFactory('FOA2')(output_trans, embpairs_trans[:,1,:], embpairs_trans[:,0,:],
                        metric=('C' if 'C' in transfer['transfer'] else 'E'))
                if 'SP' not in atype:
                    print('* <transfer> Adversarial Sample', 'FOA:Accu=', acc_trans)
                else:
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm='+',
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                    print('* <transfer> Adversarial Sample', 'FOA:Accu=', acc_trans,
                            'GT.R@mean=', rank_sp_trans)
        elif (atype in ['FOA', 'SPFOA']) and (M is not None) and (M > 2):
            # -- [eval adv] ranking attack, M>2
            loss, tau = LossFactory('FOAX')(output, embpairs, metric=metric)
            #== Semantic preserving? (SP)
            if 'SP' in atype:
                loss_sp, rank_sp = LossFactory('QA')(output, embgts, candi[0],
                        metric=metric, pm='+', dist=dist, cidx=mgtruth)
                loss = loss + XI * loss_sp
                prankgt_adv = rank_sp #/ candi[0].size(0)

            correct_adv = tau * output.shape[0] / 100.
            loss_adv = loss.clone().detach()
            rankup = 0
            if verbose:
                if 'SP' not in atype:
                    print('* Adversarial Example', 'loss=', loss.item(),
                            'FOA:tau=', tau, 'embShift=', embshift)
                else:
                    print('* Adversarial Example', 'loss=', loss.item(), 'where loss_sp=', loss_sp.item(),
                            'FOA:tau=', tau, 'GT.R@mean=', rank_sp, 'embShift=', embshift)
            # <transfer>
            if transfer is not None:
                _, tau_trans = LossFactory('FOAX')(output_trans, embpairs_trans,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'))
                if 'SP' not in atype:
                    print('* <transfer> Adversarial Sample', 'FOA:tau=', tau_trans)
                else:
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm='+',
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                    print('* <transfer> Adversarial Sample', 'FOA:tau=', tau_trans,
                            'GT.R@mean=', rank_sp_trans)
        elif (atype in ['QA', 'SPQA']) and (M is not None):
            # evaluate the adversarial examples
            if 'SP' in atype:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss_sp, rank_sp = LossFactory('QA')(output, embgts, candi[0],
                        metric=metric, pm='+', dist=dist, cidx=mgtruth)
                loss = loss_qa + XI * loss_sp
            else:
                loss_qa, rank_qa = LossFactory('QA')(output, embpairs, candi[0],
                        metric=metric, pm=pm, dist=dist, cidx=msample)
                loss = loss_qa
            mrank = rank_qa / candi[0].shape[0]
            correct_adv = mrank * output.shape[0] / 100.
            loss_adv = loss.clone().detach()
            rankup = (correct_adv - correct_orig) * 100. / output.shape[0]
            if 'SP' in atype:
                mrankgt = rank_sp / candi[0].shape[0]
                sp_adv = mrankgt * output.shape[0] / 100.
                prankgt_adv = mrankgt
            if verbose:
                if 'SP' in atype:
                    print('* Adversarial Sample', 'loss=', loss.item(),
                            f'SPQA{pm}:rank=', mrank,
                            f'SPQA{pm}:GTrank=', mrankgt,
                            'embShift=', embshift)
                else:
                    print('* Adversarial Sample', 'loss=', loss.item(),
                            f'QA{pm}:rank=', mrank, 'embShift=', embshift)
            # <transfer>
            if transfer is not None:
                _, rank_qa_trans = LossFactory('QA')(output_trans, embpairs_trans,
                        transfer['candidates'][0], pm=pm,
                        metric=('C' if 'C' in transfer['transfer'] else 'E'),
                        dist=dist_trans, cidx=msample)
                if 'SP' in atype:
                    _, rank_sp_trans = LossFactory('QA')(output_trans, embgts_trans,
                            transfer['candidates'][0], pm=pm,
                            metric=('C' if 'C' in transfer['transfer'] else 'E'),
                            dist=dist_trans, cidx=mgtruth)
                if 'SP' not in atype:
                    print('* <transfer> Adversarial Sample',
                            f'QA{pm}:rank=', rank_qa_trans / candi[0].shape[0])
                else:
                    prank_trans = rank_qa_trans / candi[0].size(0)
                    print('* <transfer> Adversarial Sample',
                            f'SPQA{pm}:rank=', prank_trans,
                            f'SPQA{pm}:GTrank=', rank_sp_trans / candi[0].shape[0])
        else:
            raise Exception("Unknown attack")
    prankgt_orig = locals().get('prankgt_orig', -1.)
    prankgt_adv = locals().get('prankgt_adv', -1.)
    prank_trans = locals().get('prank_trans', -1.)
    # >>>>>>>>>>>>>>>>>>>>>> ADVERSARIAL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # -- [report] statistics and optionally visualize
    if verbose:
        if normimg:
            r = r.mul(IMstd[:,None,None])
        cprint(' '.join(['r>', 'Min', '%.3f'%r.min().item(),
                'Max', '%.3f'%r.max().item(),
                'Mean', '%.3f'%r.mean().item(),
                'L0', '%.3f'%r.norm(0).item(),
                'L1', '%.3f'%r.norm(1).item(),
                'L2', '%.3f'%r.norm(2).item()]), 'grey')

    # historical burden, return data/stat of the current batch
    return xr, r, (output_orig, output), \
            (loss_orig, loss), (
                    [correct_orig, None],
                    [correct_adv, rankup, embshift, prankgt_adv, prank_trans]
                    )
