'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import os
import sys
import requests
import time
import hashlib
import pickle
import base64
import json
#import wx_sdk
from typing import *
import io
import numpy as np
from PIL import Image
import torch as th
from termcolor import cprint, colored
from torchvision.transforms import functional as transfunc
import requests
from tqdm import tqdm
import pylab as lab
from scipy.stats import kendalltau
import rich
c = rich.get_console()

URI_BING = 'https://api.bing.microsoft.com/v7.0/images/visualsearch'
SUBSCRIPTION_KEY = '<FILL-IN-YOUR-OWN-KEY>'


def bingQuery(im: any, id: int, verbose: bool = True, topK: int = 50, cache: bool = True) -> list:
    '''
    https://docs.microsoft.com/en-us/bing/search-apis/bing-visual-search/quickstarts/rest/python
    '''
    headers = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
            'Content-Type': 'multipart/form-data; boundary=boundary_1234-abcd'}

    if im is None:
        raise ValueError('needs an image')
    elif isinstance(im, str):
        if not os.path.exists(im):
            raise Exception(f'file {im} does not exist')
        file = {'image': ('myfile', open(im, 'rb'))}
    elif isinstance(im, np.ndarray):
        assert(len(im.shape) == 3)
        if im.shape[0] == 3 and im.shape[1] == 224 and im.shape[2] == 224:
            im_arr = np.ascontiguousarray(im.transpose((1,2,0)))
            im_arr = (im_arr * 255).astype(np.uint8)
            image = Image.fromarray(im_arr, mode='RGB')
        elif im.shape[2] == 3 and im.shape[0] == 224 and im.shape[1] == 224:
            raise NotImplementedError
        else:
            raise ValueError('neither CHW nor HWC image???')
        buf = io.BytesIO()
        image.save(buf, 'png', quality=99)
        buf.seek(0)
        content = buf.read()
        buf.close()
        file = {'image': ('myfile', content)}
    elif isinstance(im, th.Tensor):
        return bingQuery(im.clone().detach().cpu().squeeze().contiguous().numpy(),
                id=id, verbose=verbose)
    else:
        raise TypeError('illegal image type')

    def print_json(obj):
        print(json.dumps(obj, sort_keys=True, indent=2, separators=(',', ': ')))

    def hash_imgid(imgid: str):
        return int('0x' + imgid[-6:], base=16)

    try:
        #res = json.load(open('cache-cloth2.png-test.json', 'rt'))
        res = requests.post(URI_BING, headers=headers, files=file)
        res.raise_for_status()
        res = res.json()
        #print_json(res.json())
    except Exception as ex:
        raise ex

    if cache:
        if not os.path.exists('__bing_cache__'):
            os.mkdir('__bing_cache__')
        #with open(f'__bing_cache__/{os.path.basename(im)}-{id}.json', 'at') as f:
        with open(f'__bing_cache__/{id}.json', 'at') as f:
            f.write(json.dumps(res))

    data = list(filter(lambda x: x['actionType'] == 'VisualSearch',
        res['tags'][0]['actions']))[0]
    list_of_items = data['data']['value']
    if len(list_of_items) < topK:
        raise Exception('no enough retrieval results')
    list_of_ids = [x['imageId'] for x in list_of_items]
    ids = [hash_imgid(x) for x in list_of_ids[:topK]]
    #print(ids)
    # list of candidate indeces
    return ids


class BingModel(object):
    '''
    Encapsule Bing Visual Search as a black box model.

    Sometimes segfault may happen if you toggle USE_RUST_KERNEL=1
    '''

    def __init__(self, canseek: int =50):
        assert(canseek > 0)
        assert(canseek < 100)
        self.canseek = canseek
        self.xcs = th.tensor([])
        self.model = th.nn.Sequential()

    def __call__(self, query, *, id=str(time.time()), verbose=True) -> th.Tensor:
        if query.shape[0] == 1:
            print('  ', end='')
        print(f'BingModel.__call__: query.shape = {query.shape}; id = {id}')
        assert(isinstance(query, th.Tensor))
        with th.no_grad():

            if len(query.shape)==4 and query.shape[0] == 1:
                # single sample
                idlist = bingQuery(query, id=id, topK=self.canseek, verbose=verbose)
                ret = th.LongTensor(idlist), th.zeros(self.canseek)
                print(f'  [ok] BingModel.__call__: query.shape = {query.shape}; id = {id}')
                return ret

            elif len(query.shape)==4 and query.shape[0] > 1:
                # a batch of samples
                idlists = []
                for i in range(query.shape[0]):
                    idlist, _ = self(query[i].unsqueeze(0), id=f'{id}x{i}', verbose=verbose)
                    idlists.append(idlist)
                # post-processing
                maxlen = max(len(x) for x in idlists)
                for i in range(len(idlists)):
                    while len(idlists[i]) < maxlen:
                        tmp = idlists[i].tolist()
                        tmp.append(idlists[i][-1])
                        idlists[i] = th.LongTensor(tmp)
                idlists = th.stack(idlists)
                ret = idlists, th.zeros(query.shape[0], self.canseek)
                print(f'[ok] BingModel.__call__: query.shape = {query.shape}; id = {id}')
                return ret
            else:
                raise ValueError(f'problematic query shape {query.shape}')


if __name__ == '__main__':
    bingQuery('cloth2.png', id='test', verbose=True)
