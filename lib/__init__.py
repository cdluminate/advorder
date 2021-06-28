'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
# This file imports models that can be specified by "Train.py -M"
# Note, common abbr: C(osine) E(uclidean)

# fashion-mnist (fa)
from . import faC_c2f2
from . import faE_c2f2
from . import faC_lenet
from . import faC_res18

# stanfard online product (sop)
from . import sopE_res50
from . import sopE_res18
from . import sopE_dense121
from . import sopE_mnas

# reorder attack
from . import reorder

# practical reorder attack
from . import snapshop
from . import bing
