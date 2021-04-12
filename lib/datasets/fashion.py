'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import gzip
import numpy as np
import torch as th
import torch.utils.data
from collections import defaultdict
import random


class FashionPair(th.utils.data.Dataset):
    def __init__(self, path, kind='train'):
        assert(kind in ('train', 't10k', 'test'))
        if kind == 'test': kind = 't10k'
        self.kind = kind
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8).reshape(-1)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        self.labels = th.from_numpy(labels)
        self.images = th.from_numpy(images).view(-1, 1, 28, 28) / 255.0
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index: int):
        if index >= len(self.labels):
            raise IndexError
        anchor = self.images[index]
        label = self.labels[index]
        if self.kind == 'train':
            pos_indeces = th.where(self.labels == label)[0]
            positive = self.images[pos_indeces[np.random.randint(0, len(pos_indeces))]]
            return th.stack([anchor, positive]), label
        else:
            return anchor, label

def FashionPairLoader(path, batchsize, kind='train'):
    assert(kind in ('train', 't10k', 'test'))
    dataset = FashionTriplet(path, kind)
    if kind == 'train':
        return th.utils.data.DataLoader(dataset, batch_size=batchsize,
                shuffle=True, num_workers=8, pin_memory=True)
    else:
        return th.utils.data.DataLoader(dataset, batch_size=batchsize,
                shuffle=False, num_workers=2, pin_memory=True)


def test_FashionTriplet():
    dset = FashionTriplet(os.path.expanduser('~/.torch/FashionMNIST/raw'), kind='train')
    x, y = dset[0]
    print(x.shape, y.shape)

    dloader = FashionTripletLoader(os.path.expanduser('~/.torch/FashionMNIST/raw'), 2, kind='train')
    for x in dloader:
        print(x[0].shape, x[1].shape)


class FashionMNISTTriplet(th.utils.data.Dataset):
    def __init__(self, path, kind='train'):
        assert(kind in ('train', 'test', 't10k'))
        if kind == 'test': kind = 't10k'
        self.kind = kind
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8).reshape(-1)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        self.labels = th.from_numpy(labels)
        self.labelset = set(list(labels))
        self.images = th.from_numpy(images).view(-1, 1, 28, 28) / 255.0
        self.label2coll = defaultdict(list)
        for (i, lb) in enumerate(self.labels):
            self.label2coll[lb.item()].append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index: int):
        if index >= len(self.labels):
            raise IndexError
        anchor = self.images[index]
        label = self.labels[index]
        if self.kind == 'train':
            posidx = random.choice(list(self.label2coll[label.item()]))
            posimg = self.images[posidx]
            negidx = random.choice(self.label2coll[
                random.choice(list(self.labelset - {label.item()}))])
            negimg = self.images[negidx]
            assert(label.item() == self.labels[posidx].item())
            assert(label.item() != self.labels[negidx].item())
            return th.stack([anchor, posimg, negimg]), label
        else:
            return anchor, label

def FashionMNISTTripletLoader(path, batchsize, kind='train'):
    assert(kind in ('train', 'test', 't10k'))
    dataset = FashionMNISTTriplet(path, kind)
    if kind == 'train':
        return th.utils.data.DataLoader(dataset, batch_size=batchsize,
                shuffle=True, num_workers=8, pin_memory=True)
    else:
        return th.utils.data.DataLoader(dataset, batch_size=batchsize,
                shuffle=False, num_workers=4, pin_memory=True)


class FashionQuadrilateral(th.utils.data.Dataset):
    def __init__(self, path, kind='train'):
        assert(kind in ('train', 'test', 't10k'))
        if kind == 'test': kind = 't10k'
        self.kind = kind
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8).reshape(-1)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        self.labels = th.from_numpy(labels)
        self.labelset = set(list(labels))
        self.images = th.from_numpy(images).view(-1, 1, 28, 28) / 255.0
        self.label2coll = defaultdict(list)
        for (i, lb) in enumerate(self.labels):
            self.label2coll[lb.item()].append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index: int):
        if index >= len(self.labels):
            raise IndexError
        anchor = self.images[index]
        label = self.labels[index]
        if self.kind == 'train':
            posidx = random.choice(list(self.label2coll[label.item()]))
            posimg = self.images[posidx]
            neglbl = random.choice(list(self.labelset - {label.item()}))
            negidx = random.choice(self.label2coll[neglbl])
            negimg = self.images[negidx]
            negani = random.choice(self.label2coll[neglbl])
            neganc = self.images[negani]
            assert(label.item() == self.labels[posidx].item())
            assert(label.item() != self.labels[negidx].item())
            return th.stack([anchor, posimg, negimg, neganc]), label
        else:
            return anchor, label


def FashionQuadrilateralLoader(path, batchsize, kind='train'):
    assert(kind in ('train', 'test', 't10k'))
    dataset = FashionQuadrilateral(path, kind)
    if kind == 'train':
        return th.utils.data.DataLoader(dataset, batch_size=batchsize,
                shuffle=True, num_workers=8, pin_memory=True)
    else:
        return th.utils.data.DataLoader(dataset, batch_size=batchsize,
                shuffle=False, num_workers=4, pin_memory=True)



def get_loader(path: str, batchsize: int, kind='train'):
    """
    Load MNIST data and turn them into dataloaders
    """
    return FashionMNISTTripletLoader(path, batchsize, kind)

def get_label(n):
    '''
    Get the label list
    '''
    return """
    T-shirt/top
    Trouser
    Pullover
    Dress
    Coat
    Sandal
    Shirt
    Sneaker
    Bag
    Ankle boot
    """.split()[n]

def test_FashionMNISTTriplet():
    dset = FashionMNISTTriplet(os.path.expanduser('~/.torch/FashionMNIST/raw'), kind='train')
    x, y = dset[0]
    print(x.shape, y.shape)

    dloader = FashionMNISTTripletLoader(os.path.expanduser('~/.torch/FashionMNIST/raw'), 5, kind='train')
    for x in dloader:
        print(x[0].shape, x[1].shape)


if __name__ == '__main__':
    test_FashionMNISTTriplet()
