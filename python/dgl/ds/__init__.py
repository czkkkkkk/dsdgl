from .._ffi.function import _init_api

import torch as th

from .sampling import sample_neighbors

def init(rank, world_size):
    _CAPI_DGLDSInitialize(rank, world_size)

_init_api("dgl.ds")