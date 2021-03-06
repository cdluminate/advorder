'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import torch as th
import collections
import math
import yaml
from . import datasets, common
import torchvision as vision
from tqdm import tqdm
import numpy as np
from termcolor import cprint, colored
from . import utils
from . import rankingmodel


class Model(rankingmodel.Model):
    """
    Res18 + SOP dataset
    http://cvgl.stanford.edu/projects/lifted_struct/
    """
    def to(self, device):
        self.device = device
        return self._apply(lambda x: x.to(device))

    def __init__(self, finetune=True):
        super(Model, self).__init__()
        self.finetune = finetune
        if not finetune:
            cprint('Note, setting pretrain=False for ResNet18 as requested', 'red')
        self.resnet = vision.models.resnet18(pretrained=True if finetune else False)
        #self.resnet = th.nn.DataParallel(self.resnet, dim=0)
        self.resnet.fc = th.nn.Identity()
        self.metric = 'E'
        self.cachepath = 'trained/__cache__.embs.sopE_res18.th'

    def forward(self, x, *, l2norm=False):
        '''
        Input[x]: batch of images [N, 3, 224, 224]
        Output: representations [N, d]
        '''
        # -1, 3, 224, 224
        x = utils.renorm(x)
        x = self.resnet(x)
        # -1, ?
        if l2norm:
            x = x / x.norm(2, dim=1, keepdim=True).expand(*x.shape)
        return x

    def loss(self, x, y, *, marginE=1.0):
        '''
        Input[x]: image triplets in shape [3N, 3, 224, 224]
        Input[y]: not used
        '''
        x = x.to(self.device).view(-1, 3, 224, 224)
        y = y.to(self.device).view(-1) # y is not used here
        output = self.forward(x)
        #[ triplet ]
        loss = th.nn.functional.triplet_margin_loss(
                    output[0::3], output[1::3], output[2::3],
                    margin=marginE, p=2, reduction='mean')
        #[ quad : not better]
        #Floss = th.nn.functional.triplet_margin_loss(
        #        output[0::4], output[1::4], output[2::4],
        #        margin=marginE, p=2, reduction='mean')
        #Bloss = th.nn.functional.triplet_margin_loss(
        #        output[3::4], output[2::4], output[1::4],
        #        margin=marginE, p=2, reduction='mean')
        #loss = Floss + Bloss

        # [conjugate rhomboid]
        #__triplet = th.nn.functional.triplet_margin_loss
        #nf = __triplet(output[0::4], output[1::4], output[2::4])
        #nb = __triplet(output[3::4], output[2::4], output[1::4])
        #cf = __triplet(output[1::4], output[0::4], output[3::4])
        #cb = __triplet(output[2::4], output[3::4], output[0::4])
        #loss = nf + nb + cf + cb

        return output, loss

    def loss_adversary(self, x, y, *, eps=0.0, maxiter=10, hard=False):
        '''
        Train the network with a PGD adversary
        This is the defense method used in ECCV2020 paper
        M. Zhou, et al., "Adversarial Ranking Attack and Defense".
        '''
        images = x.clone().detach().to(self.device).view(-1, 3, 224, 224)
        labels = y.to(self.device).view(-1)
        images_orig = images.clone().detach()
        images.requires_grad = True

        # first evaluation
        with th.no_grad():
            output, loss_orig = self.loss(images, labels)
            output_orig = output.clone().detach()
            output_orig_nodetach = output

        # start PGD attack, we only move the anchor point
        alpha = float(min(max(eps/10., 1./255.), 0.01))  # PGD hyper-parameter
        maxiter = int(min(max(10, 2*eps/alpha), 30))  # PGD hyper-parameter

        # PGD with/without random init?
        if int(os.getenv('RINIT', 0))>0:
            images = images + eps*2*(0.5-th.rand(images.shape)).to(images.device)
            images = th.clamp(images, min=0., max=1.)
            images = images.detach()
            images.requires_grad = True

        for iteration in range(maxiter):
            self.train()
            optim = th.optim.SGD(self.parameters(), lr=1.)
            optimx = th.optim.SGD([images], lr=1.)
            optim.zero_grad(); optimx.zero_grad()

            # do we only attack the anchor?
            USE_STRIPE=False # STRIPE: a~, p, n; NO-STRIPE: a~,p~,n~
            stripe = 3 if images.shape[0]%3==0 else 2
            if USE_STRIPE and self.metric == 'C':
                output = self.forward(images[::stripe], l2norm=True)
                distance = 1 - th.mm(output, output_orig[::stripe].t())
                loss = -distance.trace() # gradient ascent on trace, i.e. diag.sum
            elif (not USE_STRIPE) and self.metric == 'C':
                output = self.forward(images, l2norm=True)
                distance = 1 - th.mm(output, output_orig.t())
                loss = -distance.trace()
            elif USE_STRIPE and self.metric == 'E':
                output = self.forward(images[::stripe], l2norm=False)
                distance = th.nn.functional.pairwise_distance(
                        output, output_orig[::stripe], p=2)
                loss = -distance.sum()
            elif (not USE_STRIPE) and self.metric == 'E':
                output = self.forward(images, l2norm=False)
                distance = th.nn.functional.pairwise_distance(
                        output, output_orig, p=2)
                loss = -distance.sum()
            loss.backward()

            # the update and projection
            images.grad.data.copy_(alpha * th.sign(images.grad))
            optimx.step()
            images = th.min(images, images_orig + eps)
            images = th.max(images, images_orig - eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            #print('> Internal PGD loop [', iteration, ']', 'loss=', loss.item())
        optim = th.optim.SGD(self.parameters(), lr=1.)
        optimx = th.optim.SGD([images], lr=1.)
        optim.zero_grad(); optimx.zero_grad()
        images.requires_grad = False

        # forward the adversarial example
        #== min(Trip(max(ES))) defense
        output, loss_adv = self.loss(images, labels)
        print('* Orig loss', '%.5f'%loss_orig.item(), '  |  ',
                '[Adv loss]', '%.5f'%loss_adv.item())
        return output, loss_adv

    def report(self, epoch, iteration, total, output, labels, loss):
        pdistAP = th.nn.functional.pairwise_distance(output[0::3],output[1::3])
        pdistAN = th.nn.functional.pairwise_distance(output[0::3],output[2::3])
        #pdistAP = th.nn.functional.pairwise_distance(output[0::4],output[1::4])
        #pdistAN = th.nn.functional.pairwise_distance(output[0::4],output[2::4])
        ineqacc = (pdistAP < pdistAN).float().mean()
        print(f'Eph[{epoch}][{iteration}/{total}]',
                collections.namedtuple('Res', ['loss', 'ineqacc'])(
                    '%.4f'%loss.item(), '%.4f'%ineqacc.item()))

    def validate(self, dataloader):
        self.eval()

        # gather the embedding vectors
        allRepr, allLabel = self.compute_embedding(dataloader, l2norm=False, device=self.device)
        allLabel = allLabel.cpu().squeeze().numpy()

        # calculate the per-example recall
        r_mean = []
        r_1, r_10 = [], []
        with th.no_grad():
            for i in tqdm(range(allRepr.shape[0])):
                xq = allRepr[i].view(1, -1)  # [1,512]
                yq = allLabel[i]
                mpdist = (allRepr - xq).norm(dim=1).cpu().numpy().squeeze()
                agsort = mpdist.argsort()[1:]
                rank = np.where(allLabel[agsort] == yq)[0].min()
                r_mean.append(rank)
                r_1.append(rank == 0)
                r_10.append(rank < 10)
        r_mean = np.mean(r_mean)
        r_1 = np.mean(r_1)
        r_10 = np.mean(r_10)

        return f'r@mean {r_mean:.1f} (/{allRepr.size(0)}) r@1 {r_1:.3f} (/1)' + \
                f'r@10 {r_10:.3f} (/1)'

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'train':
            return datasets.sop.get_loader(
                    os.path.expanduser(config['sop']['path']),
                    batchsize, 3, 'train')
        else:
            return datasets.sop.get_loader(
                    os.path.expanduser(config['sop']['path']),
                    batchsize, 1, 'test')

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.device
        dconf['normimg'] = False # XXX: False because we do renorm on forward
        dconf['device'] = self.device
        dconf['metric'] = self.metric
        #if 'PGD-UT' in att:
        #    return common.QA_PGD_UT(self, loader, dconf=dconf, verbose=verbose)
        return common.rank_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)

'''
* What should be used for the FC part in the embedding resnet?
self.resnet.fc = th.nn.Identity()
Validate[41] r@mean 126.6 (/60502) r@1 0.668 (/1)r@10 0.828 (/1) r@100 0.925 (/1) r@1000 0.978 (/1)
(baseline)

self.resnet.fc = th.nn.Sequential(th.nn.Linear(512, 512))
Validate[41] r@mean 130.3 (/60502) r@1 0.653 (/1)r@10 0.816 (/1) r@100 0.922 (/1) r@1000 0.978 (/1)
(worse)

self.resnet.fc = th.nn.Sequential(th.nn.ReLU(), th.nn.Linear(512, 512))
Validate[41] r@mean 131.8 (/60502) r@1 0.654 (/1)r@10 0.817 (/1) r@100 0.922 (/1) r@1000 0.977 (/1)
(worse)
* conclusion: Identity()

* Quad: R@1 0.645 R@10 0.812 (worse)

* Conjugate Rhomboid: r@mean 115.7 r@1 0.685 r@10 0.841
'''
