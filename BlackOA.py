#!/usr/bin/env python3
'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import sys, os, yaml, re, json, csv
import numpy as np, torch as th
from lib import reorder
import lib
import argparse, collections
from termcolor import cprint, colored
from scipy.stats import kendalltau
import statistics
try:
    import apex
except ImportError:
    apex = False
    pass


def BlackAttack(argv):
    '''
    Attack a pre-trained model
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device', type=str, default='cuda' if th.cuda.is_available() else 'cpu')
    ag.add_argument('-A', '--attack', type=str, required=True,
            choices=['RandSearch', 'PSO', 'Batk', 'Natk', 'Uatk', 'NES', 'BSch', 'SPSA'])
    ag.add_argument('-e', '--epsilon', default=4./255., type=float)
    ag.add_argument('-M', '--model', type=str, required=True)
    ag.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    ag.add_argument('-b', '--batchsize', type=int, default=100)
    ag.add_argument('-k', '--topk', type=int, default=5)
    ag.add_argument('-p', '--probe', type=float, default=1e3)
    ag.add_argument('-N', '--numquery', type=int, default=100)
    ag.add_argument('-c', '--canseek', type=int, default=-1)
    ag.add_argument('-V', '--visualize', action='store_true')
    ag.add_argument('-P', '--parallel', type=int, default=1)
    ag.add_argument('--scorelog', type=str, default='')
    ag.add_argument('--evaluate', action='store_true', help='evaluate recall performance before attacking')
    ag.add_argument('--vdist', action='store_true', help='show dist in vis result')
    ag.add_argument('--vinter', action='store_true', help='interactive vis mode')
    ag = ag.parse_args(argv)
    if ag.vdist or ag.vinter: ag.visualize = True
    cprint(json.dumps(vars(ag), indent=4), 'yellow')

    # Process the arguments
    Mname, Mpath = ag.model, 'trained/' + ag.model + '.sdth'
    assert(ag.topk > 1)
    if ag.visualize:
        import pylab as lab

    # Load the test dataset
    print('>>> Loading dataset ...', end=' ')
    loader_test = getattr(lib.reorder, Mname).getloader('test', ag.batchsize)
    print('| Testing dataset size =', len(loader_test.dataset))

    # Load the target model
    cprint(f'Setting up the {ag.model} Model')
    print(f'>>> Loading black-box target {Mname} model from:', Mpath)
    model = getattr(lib.reorder, Mname)(loader_test, canseek=ag.canseek, isadataset=True, device=ag.device)
    print(model.model)
    if apex:
        pass
        #cprint('! Using APEX AMP for inference.', 'yellow')
        #model.model = apex.amp.initialize(model.model, opt_level='O0').to(ag.device)
        # NOTE: sadly the model is slower with AMP (O1)
    model.model.eval()
    if ag.evaluate:
        print(colored('> Testing Recall Performance ...', 'red', 'on_white'),
                model.model.validate(loader_test))

    # Start attacking
    cprint(f'>_< Starting {ag.attack} Attack with Epsilon = {ag.epsilon:.3f}',
            'red', None, ['bold', 'underline'])

    if ag.scorelog:
        csvf = open(ag.scorelog, 'wt')
        csvw = csv.writer(csvf, delimiter=' ')
    rt_scores, rt_mranks, rt_aux_scores = [], [], []
    for (i, pack) in enumerate(loader_test.dataset):
        # traverse the whole test dataset, one-by-one.
        if i > ag.numquery:
            break
        query, label = pack[0].unsqueeze(0).to(ag.device), pack[1]
        argsort, dist = model(query.to(ag.device))
        orig_argsort = argsort.clone().detach()
        orig_dist = dist.clone().detach()
        rperm = th.randperm(ag.topk)
        otopk = argsort[:len(rperm)]
        rtopk = otopk[rperm]
        cprint(f'| Query ID {i:6d} |', 'red', 'on_white')
        print('| otopk =', otopk.cpu())
        print('| rperm =', rperm.cpu())
        print('| rtopk =', rtopk.cpu())
        qr, r, score, mrank, aux = getattr(reorder, ag.attack)(
                model, query, rperm,
                eps=ag.epsilon,
                maxprobe=ag.probe,
                parallel=ag.parallel, verbose=True)
        argsort, dist = model(qr)
        cprint(f'|                 >', 'red', 'on_white', end=' ')
        print(f'argsort[:topk]={argsort[:len(rperm)].cpu()}')
        if ag.scorelog:
            # [num of query, score]
            csvw.writerow([aux[1], score])
        rt_scores.append(score)
        rt_mranks.append(mrank)
        rt_aux_scores.append(aux[0])
        if (i+1)%10 == 0:
            marker = f'RunningAverage(K={ag.topk}|nQuery {ag.numquery}|seeK {ag.canseek}|e {ag.epsilon}|Probe {ag.probe}|Atk {ag.attack})'
            cprint(marker, 'white', 'on_blue')
            print('MEAN orig score', statistics.mean(rt_aux_scores))
            print('MEAN score', statistics.mean(rt_scores))
            print('MEAN mrank', statistics.mean(rt_mranks))
        # BEGIN visualization
        if ag.visualize:
            scorerperm = kendalltau(np.arange(len(rperm)), rperm.numpy()).correlation
            if scorerperm > 0.5:
                print(f'! Auto Skip visualization because rperm {rperm} too simple')
                continue
            scorethresh = {5: 0.75, 10: 0.60, 25: 0.50}
            if score < scorethresh[ag.topk]:
                print(f'! Auto Skip visualization because score = {score} < {scorethresh[ag.topk]}')
                continue
            if (rperm == th.arange(ag.topk).t()).sum() == ag.topk:
                print(f'! Auto Skip visualization because rperm == arange')
            autosize = {5: (12*1.5, 5*1.5),
                    10: (22*1.5, 5*1.5),
                    25: (52*1.5, 5*1.5)}
            lab.figure(figsize=autosize[ag.topk])
            if len(query.cpu().squeeze().numpy().shape) == 2:
                fs = 12  # fontsize
            else:
                fs = 8   # fontsize
            showdist = ag.vdist
            def lab_imshow(im):
                if len(im.shape) == 2: # MNIST-Like, gray
                    if im.min() < 0.0: # perturbation
                        lab.imshow(im, cmap='gray', vmin=-1, vmax=1); lab.axis(False)
                    else: # image itself
                        lab.imshow(im, cmap='gray');
                        if not showdist: lab.axis(False)
                elif len(im.shape) == 3 and (im.shape[0] == 3): # imagenet-like
                    if im.min() < 0.0: # perturbation
                        lab.imshow(im.transpose((1,2,0)) / 2 + 0.5); lab.axis(False)
                    else: # image itself
                        lab.imshow(im.transpose((1,2,0)));
                        if not showdist: lab.axis(False)
                else:
                    raise ValueError
            if ag.canseek < 0 or ag.canseek >= 2 * ag.topk:
            # BEGIN case canseek infty
                # line1: draw the query
                lab.subplot(4, 2+ag.topk*2, 3)
                lab_imshow(query.cpu().squeeze().numpy())
                lb = label.item()
                lab.title(f'ᴵᴰ{i} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                lab.subplot(4, 2+ag.topk*2, 2+ag.topk*2 + 1)
                lab_imshow(query.cpu().squeeze().numpy())
                lab.title(f'ᴵᴰ{i} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                lab.subplot(4, 2+ag.topk*2, 5)
                lab_imshow(r.cpu().squeeze().numpy())
                lab.subplot(4, 2+ag.topk*2, 7)
                lab_imshow(qr.cpu().squeeze().numpy())
                lab.title(f'ᴵᴰ{i}ᴬᴰⱽ', fontsize=fs)
                lab.subplot(4, 2+ag.topk*2, 6+ag.topk*6 + 1)
                lab_imshow(qr.cpu().squeeze().numpy())
                lab.title(f'ᴵᴰ{i}ᴬᴰⱽ', fontsize=fs)
                # line2+4: draw the orig and adv ranking
                for vi in range(ag.topk*2):
                    # line 2
                    lab.subplot(4, 2+ag.topk*2, 2+ag.topk*2 + vi+3)
                    lab_imshow(loader_test.dataset[orig_argsort[vi]][0].cpu().squeeze().numpy());
                    lb = loader_test.dataset[orig_argsort[vi]][1].item()
                    lab.title(f'ᴵᴰ{orig_argsort[vi].item()} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                    lab.xlabel(f'qDist={orig_dist[vi].item():.3f}', fontsize=4)
                    # line 4
                    lab.subplot(4, 2+ag.topk*2, 6+ag.topk*6 + vi+3)
                    lab_imshow(loader_test.dataset[argsort[vi]][0].cpu().squeeze().numpy());
                    lb = loader_test.dataset[argsort[vi]][1].item()
                    lab.title(f'ᴵᴰ{argsort[vi].item()} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                    lab.xlabel(f'qDist={dist[vi].item():.3f}', fontsize=4)
                # line3: draw the desired ranking
                for vi in range(ag.topk):
                    lab.subplot(4, 2+ag.topk*2, 4+ag.topk*4 + vi+3)
                    lab_imshow(loader_test.dataset[rtopk[vi]][0].cpu().squeeze().numpy());
                    lb = loader_test.dataset[rtopk[vi]][1].item()
                    lab.title(f'ᴵᴰ{rtopk[vi].item()} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                # draw
                if ag.vinter:
                    lab.show()
                else:
                    lab.savefig(f'{ag.attack}-k{ag.topk}c{ag.canseek}-e{ag.epsilon:.3f}-Q{i}-Sc{score:.2f}.svg')
                    cprint(f'{ag.attack}-k{ag.topk}c{ag.canseek}-e{ag.epsilon:.3f}-Q{i}-Sc{score:.2f}.svg', 'yellow', None, ['bold'])
            # END case canseek infty
            elif ag.canseek > 0 and ag.canseek < 2 * ag.topk:
            # BEGIN case canseek k
                # line1: draw the query
                lab.subplot(4, 2+ag.topk, 3)
                lab_imshow(query.cpu().squeeze().numpy())
                lb = label.item()
                lab.title(f'ᴵᴰ{i} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                lab.subplot(4, 2+ag.topk, 2+ag.topk + 1)
                lab_imshow(query.cpu().squeeze().numpy())
                lab.title(f'ᴵᴰ{i} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                lab.subplot(4, 2+ag.topk, 5)
                lab_imshow(r.cpu().squeeze().numpy())
                lab.subplot(4, 2+ag.topk, 7)
                lab_imshow(qr.cpu().squeeze().numpy())
                lab.title(f'ᴵᴰ{i}ᴬᴰⱽ', fontsize=fs)
                lab.subplot(4, 2+ag.topk, 6+ag.topk*3 + 1)
                lab_imshow(qr.cpu().squeeze().numpy())
                lab.title(f'ᴵᴰ{i}ᴬᴰⱽ', fontsize=fs)
                # line2+4: draw the orig and adv ranking
                for vi in range(ag.topk):
                    # line 2
                    lab.subplot(4, 2+ag.topk, 2+ag.topk + vi+3)
                    lab_imshow(loader_test.dataset[orig_argsort[vi]][0].cpu().squeeze().numpy());
                    lb = loader_test.dataset[orig_argsort[vi]][1].item()
                    lab.title(f'ᴵᴰ{orig_argsort[vi].item()} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                    lab.xlabel(f'qDist={orig_dist[vi].item():.3f}', fontsize=4)
                    # line 4
                    lab.subplot(4, 2+ag.topk, 6+ag.topk*3 + vi+3)
                    lab_imshow(loader_test.dataset[argsort[vi]][0].cpu().squeeze().numpy());
                    lb = loader_test.dataset[argsort[vi]][1].item()
                    lab.title(f'ᴵᴰ{argsort[vi].item()} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                    lab.xlabel(f'qDist={dist[vi].item():.3f}', fontsize=4)
                # line3: draw the desired ranking
                for vi in range(ag.topk):
                    lab.subplot(4, 2+ag.topk, 4+ag.topk*2 + vi+3)
                    lab_imshow(loader_test.dataset[rtopk[vi]][0].cpu().squeeze().numpy());
                    lb = loader_test.dataset[rtopk[vi]][1].item()
                    lab.title(f'ᴵᴰ{rtopk[vi].item()} ˡᵃᵇᵉˡ{lb}', fontsize=fs)
                # draw
                if ag.vinter:
                    lab.show()
                else:
                    lab.savefig(f'{ag.attack}-k{ag.topk}c{ag.canseek}-e{ag.epsilon:.3f}-Q{i}-Sc{score:.2f}.svg')
                    cprint(f'{ag.attack}-k{ag.topk}c{ag.canseek}-e{ag.epsilon:.3f}-Q{i}-Sc{score:.2f}.svg', 'yellow', None, ['bold'])
            # END case canseek k
            else:
                raise Exception("they way to plot in this case is undefined.")
        # END visualziation
    marker = f'FINAL(K={ag.topk}|nQuery {ag.numquery}|seeK {ag.canseek}|e {ag.epsilon}|Probe {ag.probe}|Atk {ag.attack})'
    cprint(marker, 'white', 'on_blue')
    print('MEAN orig score', statistics.mean(rt_aux_scores))
    print('MEAN score', statistics.mean(rt_scores))
    print('MEAN mrank', statistics.mean(rt_mranks))

    if ag.scorelog:
        csvf.close()


if __name__ == '__main__':
    BlackAttack(sys.argv[1:])
