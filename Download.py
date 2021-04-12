#!/usr/bin/env python3
'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import sys
import torchvision as V
from termcolor import cprint

cprint('>_< FashionMNIST', 'white', 'on_blue')
V.datasets.FashionMNIST('~/.torch/', download=True)

cprint('>_< Stanford Online Products', 'white', 'on_blue')
print('Download the dataset here: ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip')

cprint('>_< Done!', 'white', 'on_green')
