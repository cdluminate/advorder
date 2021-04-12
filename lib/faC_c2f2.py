'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import torch as th
import collections
import math
import yaml
import numpy as np
from . import datasets, common
from . import rankingmodel

class Model(rankingmodel.Model):
    """
    LeNet-like convolutional neural network for embedding
    https://github.com/zalandoresearch/fashion-mnist/blob/master/benchmark/convnet.py
    """
    def to(self, device):
        self.device = device
        return self._apply(lambda x: x.to(device))

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = th.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = th.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = th.nn.Linear(64*7*7, 1024)
        self.metric = 'C'

    def forward(self, x, *, l2norm=False):
        '''
        Input[x]: tensor in shape [N, 1, 28, 28]
        Output: tensor in shape [N, d]
        '''
        assert(len(x.shape) == 4)
        # -1, 1, 28, 28
        x = th.nn.functional.relu(self.conv1(x))
        # -1, 32, 28, 28
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 32, 14, 14
        x = th.nn.functional.relu(self.conv2(x))
        # -1, 64, 14, 14
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 64, 7, 7
        x = x.view(-1, 64*7*7)
        # -1, 64*7*7
        x = self.fc1(x)
        # -1, 1024
        if l2norm:
            x = th.nn.functional.normalize(x, p=2, dim=1)
        return x

    def loss_adversary(self, x, y, *, eps=0.0, maxiter=10, hard=False, marginC=0.2, marginE=1.0):
        '''
        Train the network with a PGD adversary
        '''
        raise NotImplementedError
        images = x.to(self.fc1.weight.device)
        labels = y.to(self.fc1.weight.device).view(-1)
        images_orig = images.clone().detach()
        images.requires_grad = True

        # first forwarding
        with th.no_grad():
            output, loss_orig = self.loss(images, labels)
            if self.metric == 'C':
                output /= output.norm(2, dim=1, keepdim=True).expand(*output.shape)
            elif self.metric == 'E':
                pass
            output_orig = output.clone().detach()
            output_orig_nodetach = output

        # start PGD attack
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

            if self.metric == 'C':
                output = self.forward(images, l2norm=True)
                distance = 1 - th.mm(output, output_orig.t())
                loss = -distance.trace() # gradient ascent on trace, i.e. diag.sum
            elif self.metric == 'E':
                output = self.forward(images, l2norm=False)
                distance = th.nn.functional.pairwise_distance(
                        output, output_orig, p=2)
                loss = -distance.sum()
            else:
                raise ValueError(self.metric)
            loss.backward()

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
        if False:
            #== trip-es loss
            _, loss_adv = self.loss(images_orig, labels)
            if self.metric == 'C':
                output = self.forward(images, l2norm=True)
                loss_es = (1 - th.mm(output, output_orig_nodetach.t())).trace() / output.size(0)
                loss_adv = loss_adv + 1.0 * loss_es
            elif self.metric == 'E':
                output = self.forward(images, l2norm=False)
                loss_es = th.nn.functional.pairwise_distance(output, output_orig_nodetach, p=2).mean()
                loss_adv = loss_adv + 1.0 * loss_es  # very unstable
            print('* Orig loss', '%.5f'%loss_orig.item(), '\t|\t',
                    '[Adv loss]', '%.5f'%loss_adv.item(), '\twhere loss_ES=', loss_es.item())
            return output, loss_adv
        else:
            #== min(Trip(max(ES(\tilde(a))))) loss
            output, loss_adv = self.loss(images, labels)
            print('* Orig loss', '%.5f'%loss_orig.item(), '\t|\t',
                    '[Adv loss]', '%.5f'%loss_adv.item())
            return output, loss_adv

    def report(self, epoch, iteration, total, output, labels, loss):
        #X = output / output.norm(2, dim=1, keepdim=True).expand(*(output.shape))
        #dist = 1 - X @ X.t()
        #offdiagdist = dist + math.e*th.eye(X.shape[0]).to(X.device)
        #knnsearch = labels[th.min(offdiagdist, dim=1)[1]].cpu()
        #acc = 100.*knnsearch.eq(labels.cpu()).sum().item() / len(labels)
        #print(f'Eph[{epoch}][{iteration}/{total}]',
        #        collections.namedtuple('Res', ['loss', 'r_1'])(
        #            '%.4f'%loss.item(), '%.3f'%acc))
        print(f'Eph[{epoch}][{iteration}/{total}] loss {loss.item()}')

    def validate(self, dataloader):
        self.eval()
        allFeat, allLab = self.compute_embedding(dataloader,
                l2norm=True, device=self.device)
        dist = 1 - th.mm(allFeat, allFeat.t())
        offdiagdist = dist + \
                math.e * th.eye(allFeat.shape[0]).to(allFeat.device)
        knnsearch = allLab[th.min(offdiagdist, dim=1)[1]].cpu()
        correct = knnsearch.eq(allLab.cpu()).sum().item()
        total = len(allLab)
        result = f'r@1 = {100.*correct/total} (/100)'
        return result

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)
        path = os.path.expanduser(config['fashion-mnist']['path'])
        #return datasets.fashion.FashionQuadrilateralLoader(path, batchsize, kind)
        return datasets.fashion.get_loader(path, batchsize, kind)

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.fc1.weight.device
        dconf['metric'] = self.metric
        return common.rank_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)

    def loss(self, x, y, *, marginC=0.2, hard=True):
        '''
        Input[x]: images [N, 1, 28, 28]
        Input[y]: corresponding image labels [N]
        '''
        x = x.view(-1, 1, 28, 28).to(self.fc1.weight.device)
        y = y.view(-1).to(self.fc1.weight.device).view(-1)
        output = self.forward(x, l2norm=True)

        # Performance: R@1 88.77 (/100)
        dAP = 1 - th.nn.functional.cosine_similarity(
                output[0::3], output[1::3])
        dAN = 1 - th.nn.functional.cosine_similarity(
                output[0::3], output[2::3])
        loss = (dAP - dAN + marginC).clamp(min=0.).mean()

        # Rhomboid Loss
        # Performance: R@1 89.62 (/100)
        #FdAP = 1 - th.nn.functional.cosine_similarity(output[0::4], output[1::4])
        #FdAN = 1 - th.nn.functional.cosine_similarity(output[0::4], output[2::4])
        #BdAP = 1 - th.nn.functional.cosine_similarity(output[3::4], output[2::4])
        #BdAN = 1 - th.nn.functional.cosine_similarity(output[3::4], output[1::4])
        #loss = ((FdAP - FdAN + marginC).clamp(min=0.).exp()-1).mean() + \
        #        ((BdAP - BdAN + marginC).clamp(min=0.).exp()-1).mean()

        # Conjugate Rhomboid Loss
        # Performance: R@1 90.17 (/100)
        #__cos = th.nn.functional.cosine_similarity
        #nfdAP = 1 - __cos(output[0::4], output[1::4])
        #nfdAN = 1 - __cos(output[0::4], output[2::4])
        #nbdAP = 1 - __cos(output[3::4], output[2::4])
        #nbdAN = 1 - __cos(output[3::4], output[1::4])
        #cfdAP = 1 - __cos(output[1::4], output[0::4])
        #cfdAN = 1 - __cos(output[1::4], output[3::4])
        #cbdAP = 1 - __cos(output[2::4], output[3::4])
        #cbdAN = 1 - __cos(output[2::4], output[0::4])
        #loss = ((nfdAP - nfdAN + marginC).clamp(min=0.).exp()-1).mean() + \
        #       ((nbdAP - nbdAN + marginC).clamp(min=0.).exp()-1).mean() + \
        #       ((cfdAP - cfdAN + marginC).clamp(min=0.).exp()-1).mean() + \
        #       ((cbdAP - cbdAN + marginC).clamp(min=0.).exp()-1).mean()

        return output, loss
