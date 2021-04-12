'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import torch as th

IMmean = th.tensor([0.485, 0.456, 0.406])
IMstd = th.tensor([0.229, 0.224, 0.225])

renorm = lambda im: im.sub(IMmean[:,None,None].to(im.device)).div(IMstd[:,None,None].to(im.device))
denorm = lambda im: im.mul(IMstd[:,None,None].to(im.device)).add(IMmean[:,None,None].to(im.device))
xdnorm = lambda im: im.div(IMstd[:,None,None].to(im.device)).add(IMmean[:,None,None].to(im.device))

chw2hwc = lambda im: im.transpose((0,2,3,1)) if len(im.shape)==4 else im.transpose((1,2,0))
