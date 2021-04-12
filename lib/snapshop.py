'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.

Client Library for JD SnapShop
https://neuhub.jd.com/dev/api/102
https://aidoc.jd.com/image/snapshop.html
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

URL_JD = 'https://aiapi.jd.com/jdai/snapshop'
APP_KEY = '<THIS IS SECRET, APPLY ONE BY YOURSELF AND FILL IT IN>'
SEC_KEY = '<THIS IS SECRET, APPLY ONE BY YOURSELF AND FILL IT IN>'

def JDQuery(im: Any = None, *, id:str = '', topK=50, verbose=False):
    '''
    Perform a query to JD API
    '''
    if len(APP_KEY) != 32 or len(SEC_KEY) != 32:
        raise ValueError(f'''
        {__file__}: Please request for an AppKey and a SecretKey from
            https://neuhub.jd.com/ai/api/image/snapshop
        and fill them in the APP_KEY and SEC_KEY variables, respectively.
        Without the keys it won't be possible to call the API.
        '''
        )
    print(f'Calling JDQuery with im.type={type(im)} im.shape={im.shape}')
    headers = {'Content-Type': 'text/plain'}
    tm = int(time.time()*1000 + 8*3600)  # UTC + GMT+8 offset
    query = {'appkey': APP_KEY,
            'secretkey': SEC_KEY,
            'timestamp': tm,
            'sign': hashlib.md5((SEC_KEY + str(tm)).encode()).hexdigest(),
            }
    if im is None:
        raise ValueError('Must provide an image!')
    else:
        if isinstance(im, str):
            with open(im, 'rb') as f:
                content = f.read()
            imgbase64 = base64.encodebytes(content).decode()
        elif isinstance(im, np.ndarray):
            assert(len(im.shape) == 3)  # MUST BE CHW (pth) OR HWC (pil)
            if im.shape[0] == 3 and im.shape[1] == 224 and im.shape[2] == 224:
                #image = Image.fromarray(im.transpose((1,2,0)), mode='RGB')
                #image = transfunc.to_pil_image(im, mode='RGB')
                #print(im)
                im_arr = np.ascontiguousarray(im.transpose((1,2,0)))
                im_arr = (im_arr * 255).astype(np.uint8)
                #import pylab as lab
                #lab.imshow(im_arr)
                #print('DEBUG1')
                #lab.show()
                #input('QQ1')
                #print(im_arr)
                image = Image.fromarray(im_arr, mode='RGB')
                #image.show()
                #input('QQQ2')
                #image2 = Image.fromarray(im)
                #image2.show()
                #input('QQQ11')
            elif im.shape[2] == 3 and im.shape[0] == 224 and im.shape[1] == 224:
                raise NotImplementedError
                #image = Image.fromarray(im, mode='RGB')
                #image = transfunc.to_pil_image(im, mode='RGB')
            else:
                raise ValueError('neither CHW nor HWC image???')
            #image.show()
            #input()
            #exit()
            buf = io.BytesIO()
            image.save(buf, 'png', quality=99)
            buf.seek(0)
            content = buf.read()
            buf.close()
            imgbase64 = base64.encodebytes(content).decode()
            sv = f'__snapshop__/{str(tm)}.id-{id}.png'
            with open(sv, 'wb') as f:
                f.write(content)
            cprint(f'> dumped query image to {sv}', 'yellow')
        elif isinstance(im, th.Tensor):
            return JDQuery(im.detach().clone().cpu().squeeze().contiguous().numpy(),
                    id=id, verbose=verbose)
        else:
            raise ValueError('illegal image type')
    body = {'channel_id': 'test',
            'imgBase64': imgbase64.strip().replace('\n',''),
            'topK': topK,
            }
    def flatten(d):
        return '&'.join(f'{str(k)}={str(v)}' for (k, v) in d.items())
    #raise NotImplementedError # XXX: DEBUGGING
    #print('! HEADER', headers)
    #print('! Params', query)
    #print('!   Body', flatten(body))
    if verbose: print('! POST ...', end=' ')
    res = requests.post(URL_JD, headers=headers, params=query, data=flatten(body))
    #res = wx_sdk.wx_post_req(URL_JD, query, bodyStr=flatten(body))
    if verbose: print(res.status_code)
    if not os.path.exists('__snapshop__'):
        os.mkdir('__snapshop__')
    pkl = f'__snapshop__/{str(tm)}.id-{id}.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(res, f)
    #print(res.json())
    if verbose: print(f'! pickle > {pkl}')
    if verbose: print(f'! DUMPing the ranking list')
    js = res.json()
    if 'remain' not in js:
        raise ValueError(js)
    if verbose: print(f'* META', js['msg'], 'Remain:', js['remain'])
    for (i, obj) in enumerate(js['result']['dataValue']):
        for (j, can) in enumerate(obj['sims']):
            if verbose: print(i, j, can['cid1Name'], can['cid2Name'], can['cid3Name'],
                    f'dis={can["dis"]}', f'sim={can["similarity"]}',
                    #'\n', '  ', can['skuName'],
                    #'\n',
                    '  ', can['skuId'], can['detailUrl'], sep='  ')
    return res


def detail(pkl, *, verbose=True):
    if isinstance(pkl, str):
        with open(pkl, 'rb') as f:
            pkl = pickle.load(f)
        js = pkl.json()
    else:
        js = pkl.json()
    if 'result' not in js.keys():
        print('Skipping due to invalid http response.')
        return
    for (i, obj) in enumerate(js['result']['dataValue']):
        for (j, can) in enumerate(obj['sims']):
            if verbose: print(i, j, can['cid1Name'], can['cid2Name'], can['cid3Name'],
                    f'dis={can["dis"]}', f'sim={can["similarity"]}',
                    #'\n', '  ', can['skuName'],
                    #'\n',
                    '  ', can['skuId'], can['detailUrl'], sep='  ')


def _downloader(url: str, *, CACHEDIR:str = '__snapshop__'):
    '''
    helper for downloading images
    '''
    resp = requests.get(url=url, stream=True)
    name = os.path.basename(url)
    #print(resp.headers)
    content_size = int(resp.headers['Content-Length'])//1024 + 1
    with open(os.path.join(CACHEDIR, name), 'wb') as f:
        for data in tqdm(iterable=resp.iter_content(1024), total=content_size, unit='kiB', desc=name):
            f.write(data)


def display(pkl, *, k=-1):
    if isinstance(pkl, str):
        with open(pkl, 'rb') as f:
            pkl = pickle.load(f)
        js = pkl.json()
    else:
        js = pkl.json()
    if 'result' not in js.keys():
        print('Skipping due to invalid http response.')
        return
    for (i, can) in enumerate(js['result']['dataValue'][0]['sims']):
        if k > 0 and i >= k:
            break
        print(can['skuId'], end=' ')
    print()


def Anchor(pkl, *, k=-1):
    '''
    helper for analyzing the attack results
    '''
    from glob import glob
    if isinstance(pkl, str):
        with open(pkl, 'rb') as f:
            pkl = pickle.load(f)
        js = pkl.json()
    else:
        js = pkl.json()
    if 'result' not in js.keys():
        raise KeyError("The provided anchor is invalid.")

    cprint('>_< Reference Anchor List', 'white', None, ['bold'])
    top6 = []
    colormap = {0: 'red', 1: 'yellow', 2: 'green', 3: 'cyan', 4: 'blue', 5: 'magenta'}
    for (i, can) in enumerate(js['result']['dataValue'][0]['sims']):
        if k > 0 and i >= k:
            break
        idx = can['skuId']
        if i < 6:
            top6.append(idx)
            cprint(idx, colormap[i], None, ['bold'], end='  ')
        else:
            print(idx, end='  ')
    print(); print()

    pkls = glob('__snapshop__/*.pkl')
    cprint(f'>_< Found {len(pkls)} pickle files. Start Processing ...', 'white', None, ['bold'])
    print()

    minimum = 1.0
    for (i, pk) in enumerate(sorted(pkls)):
        f = open(pk, 'rb')
        js = pickle.load(f).json()
        f.close()

        if 'result' not in js.keys():
            print(f'* Skipping invalid pkl #{i} {pk}')
            continue
        elif 'dataValue' not in js['result'].keys():
            print(f'* Skipping invalid pkl #{i} {pk}')
            continue
        else:
            cprint(f'\t>_< Listing #{i} {pk}', 'white')

        jlist = [can['skuId'] for can in js['result']['dataValue'][0]['sims']]
        jall = all(x in jlist for x in top6)
        if jall:
            cprint('GOOD', 'grey', 'on_green', end='  ')
        for (j, can) in enumerate(js['result']['dataValue'][0]['sims']):
            if k > 0 and j >= k:
                break
            idx = can['skuId']
            if idx in top6:
                cprint(idx, colormap[top6.index(idx)], None, end='  ')
            else:
                print(idx, end='  ')
        print()
        if jall:
            order = [jlist.index(x) for x in top6]
            tau = kendalltau([0,1,2,3,4,5], order).correlation
            print('    '+colored('Order After Perturbation:', 'grey', 'on_green'),
                    order, colored(str(tau), 'white', 'on_red' if tau < 0.5 else None))
            if tau < minimum:
                minimum = tau

    cprint(f'>_< Finished; Minimum = {minimum}.', 'white', None, ['bold'])
    return minimum


def visrow(pkl):
    '''
    Show the retrieved images in a row, and dump into svg images
    '''
    js = pkl.json()
    CACHEDIR = '__snapshop__'
    try:
        _ = js['result']
    except KeyError as e:
        if e.args[0] == 'result':
            print('The response is broken or invalid (out-of-limit response)')
        else:
            print('KeyError:', e)

    LskuId = []
    Lurl = []
    for (i, can) in enumerate(js['result']['dataValue'][0]['sims']):
        #c1name, c2name, c3name = (can[f'cid{str(x)}Name'] for x in (1,2,3))
        c1name, c2name, c3name = can['cid1Name'], can['cid2Name'], can['cid3Name']
        skuId, url = can['skuId'], can['imageUrl']
        print(f'Candidate #{i:3d}: {skuId:<15d} {c1name} {c2name} {c3name} {url}')
        if os.path.exists(os.path.join(CACHEDIR, os.path.basename(url))):
            pass
        else:
            _downloader(url, CACHEDIR=CACHEDIR)
        LskuId.append(skuId)
        Lurl.append(url)

    print('Drawing')
    fig = lab.figure(figsize=(36*2,1*2))
    N = 32
    for i in range(N):
        ax = lab.subplot(1, 32, i+1)
        ax.set_title(str(LskuId[i]), fontsize=5)
        im = Image.open(os.path.join(CACHEDIR, os.path.basename(Lurl[i])), 'r')
        print(im)
        lab.imshow(im)
        lab.axis(False)
    lab.show()
    fig.savefig('visrow.svg', dpi=512)


class JDModel(object):
    def __init__(self, canseek=50):
        assert(canseek > 0)
        assert(canseek < 100)
        self.canseek = canseek
        self.xcs = th.tensor([])
        self.model = th.nn.Sequential()
    def __call__(self, query, *, id=str(time.time()), verbose=True) -> th.Tensor:
        print(f'Calling JDModel.__call__ with query.shape as {query.shape}')
        assert(isinstance(query, th.Tensor))
        with th.no_grad():
            #if int(os.getenv('DEBUG', 0)) > 0:
            #    if query.shape[0] == 1:
            #        return th.LongTensor(np.random.permutation(100)[:self.canseek]), th.zeros(self.canseek)
            #    else:
            #        return th.LongTensor([np.random.permutation(100)[:self.canseek] for _ in query.shape[0]]), th.zeros((query.shape[0], self.canseek))
            if len(query.shape)==4 and query.shape[0] == 1:
                js = JDQuery(query, id=id, topK=self.canseek, verbose=verbose).json()
                idlist = [int(x['skuId']) for x in js['result']['dataValue'][0]['sims']]
                #idlist = [1 for _ in range(50)]
                return th.LongTensor(idlist), th.zeros(self.canseek)
            elif len(query.shape)==4 and query.shape[0] > 1:
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
                return idlists, th.zeros(query.shape[0], self.canseek)
            else:
                raise ValueError(f'problematic query shape {query.shape}')


if __name__ == '__main__':
    #JDQuery('test.jpg', id='')
    JDQuery('airpods.png', id='test', verbose=True)
