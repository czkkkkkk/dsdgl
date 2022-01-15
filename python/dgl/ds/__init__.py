from .._ffi.function import _init_api

import torch as th

from .sampling import sample_neighbors, rebalance_train_nids
from .to_block import to_block

def init(rank, world_size):
    _CAPI_DGLDSInitialize(rank, world_size)

_init_api("dgl.ds")