'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import torch as th
from tqdm import tqdm
from termcolor import cprint

class Model(th.nn.Module):
    '''
    Base model. May be a classification model, or a embedding/ranking model.
    '''
    def forward(self, x):
        '''
        Purely input -> output, no loss function
        '''
        raise NotImplementedError

    def loss(self, x, y, device='cpu'):
        '''
        Combination: input -> output -> loss
        '''
        raise NotImplementedError

    def loss_adversary(self, x, y, device='cpu', *, eps=0.0):
        '''
        Adversarial training: replace normal example with adv example
        https://github.com/MadryLab/mnist_challenge/blob/master/train.py
        '''
        raise NotImplementedError

    def report(self, epoch, iteration, total, output, labels, loss):
        '''
        Given the (output, loss) combination, report current stat
        '''
        raise NotImplementedError

    def validate(self, dataloader, device='cpu'):
        '''
        Run validation on the given dataset
        '''
        raise NotImplementedError

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        Load the specific dataset for the model
        '''
        raise NotImplementedError

    def attack(self, attack, loader, *, dconf, device, verbose=False):
        '''
        Apply XXX kind of attack
        '''
        raise NotImplementedError

    def compute_embedding(self, dataloader, *, l2norm=False, device='cpu', cache=False):
        '''
        Compute the embedding vectors
        '''
        if int(os.getenv('CACHE', 0)) > 0:
            # export CACHE=1 to enable
            cache = True
        self.eval()
        if cache and hasattr(self, 'cachepath') and os.path.exists(self.cachepath):
            cprint(f'! Loading embedding vectors from cache: {self.cachepath}', 'yellow')
            allEmb, allLab = th.load(self.cachepath)
            return allEmb.to(device), allLab.to(device)
        allEmb, allLab = [], []
        with th.no_grad():
            iterator = tqdm(enumerate(dataloader), total=len(dataloader))
            for iteration, (images, labels) in iterator:
                images, labels = images.to(device), labels.to(device)
                labels = labels.view(-1)
                output = self.forward(images)
                if l2norm: # do L-2 normalization
                    output = th.nn.functional.normalize(output, p=2, dim=1)
                allEmb.append(output.detach())
                allLab.append(labels)
        allEmb, allLab = th.cat(allEmb), th.cat(allLab)
        if cache and hasattr(self, 'cachepath'):
            cprint(f'! Saving embedding vectors to cache: {self.cachepath}', 'yellow')
            th.save([allEmb.detach().cpu(), allLab.detach().cpu()], self.cachepath)
        return allEmb, allLab
