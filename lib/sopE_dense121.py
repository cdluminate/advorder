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
    DenseNet121 + SOP dataset
    http://cvgl.stanford.edu/projects/lifted_struct/
    """
    def to(self, device):
        self.device = device
        return self._apply(lambda x: x.to(device))

    def __init__(self, finetune=True):
        super(Model, self).__init__()
        self.finetune = finetune
        if not finetune:
            cprint('Note, setting pretrain=False for DenseNet121 as requested', 'red')
        densenet = vision.models.densenet121(pretrained=True if finetune else False)
        self.densenet = th.nn.DataParallel(densenet, dim=0)
        self.densenet.fc = th.nn.Identity() # th.nn.Linear(2048, 2048)
        self.metric = 'E'
        self.cachepath = 'trained/__cache__.embs.sopE_dense121.th'

    def forward(self, x, *, l2norm=False):
        '''
        Input[x]: batch of images [N, 3, 224, 224]
        Output: representations [N, d]
        '''
        # -1, 3, 224, 224
        x = utils.renorm(x)
        x = self.densenet(x)
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
        # [plain triplet]
        loss = th.nn.functional.triplet_margin_loss(
                    output[0::3], output[1::3], output[2::3],
                    margin=marginE, p=2, reduction='mean')
        # [conjugate rhomboid]
        #__triplet = th.nn.functional.triplet_margin_loss
        #nf = __triplet(output[0::4], output[1::4], output[2::4])
        #nb = __triplet(output[3::4], output[2::4], output[1::4])
        #cf = __triplet(output[1::4], output[0::4], output[3::4])
        #cb = __triplet(output[2::4], output[3::4], output[0::4])
        #loss = nf + nb + cf + cb
        return output, loss

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
        r_1, r_10, r_100, r_1000 = [], [], [], []
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
                r_100.append(rank < 100)
                r_1000.append(rank < 1000)
        r_mean = np.mean(r_mean)
        r_1 = np.mean(r_1)
        r_10 = np.mean(r_10)
        r_100 = np.mean(r_100)
        r_1000 = np.mean(r_1000)

        return f'r@mean {r_mean:.1f} (/{allRepr.size(0)}) r@1 {r_1:.3f} (/1)' + \
                f'r@10 {r_10:.3f} (/1) r@100 {r_100:.3f} (/1) r@1000 {r_1000:.3f} (/1)'

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
Performance
Validate[30] r@mean 120.8 (/60502) r@1 0.672 (/1)r@10 0.831 (/1) r@100 0.928 (/1) r@1000 0.979 (/1)

batchsize=64 (4 cards) (does not worth it I guess)
Validate[41] r@mean 109.1 (/60502) r@1 0.689 (/1)r@10 0.844 (/1) r@100 0.935 (/1) r@1000 0.981 (/1)
'''
