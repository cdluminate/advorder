'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import torch as th
import collections
import math
import yaml
from . import datasets
from . import faC_c2f2
import numpy as np

class Model(faC_c2f2.Model):
    """
    LeNet-like convolutional neural network for embedding
    """
    def __init__(self):
        '''
        Caffe-LeNet
        '''
        super(Model, self).__init__()
        self.conv1 = th.nn.Conv2d(1, 20, 5, stride=1)
        self.conv2 = th.nn.Conv2d(20, 50, 5, stride=1)
        self.fc1 = th.nn.Linear(800, 500)
        #self.fc2 = th.nn.Linear(500, 10)

    def forward(self, x, *, l2norm=False):
        # -1, 1, 28, 28
        x = self.conv1(x)
        # -1, 20, 24, 24
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 20, 12, 12
        x = self.conv2(x)
        # -1, 50, 8, 8
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 50, 4, 4
        x = x.view(-1, 4*4*50)
        # -1, 800
        x = th.nn.functional.relu(self.fc1(x))
        # -1, 500
        # x = self.fc2(x)
        if l2norm:
            x = x / x.norm(2, dim=1, keepdim=True).expand(*x.shape)
        return x
