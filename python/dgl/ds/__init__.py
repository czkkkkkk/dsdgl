from .._ffi.function import _init_api

import torch as th

from .sampling import sample_neighbors
from .to_block import to_block

def init(rank, world_size):
    id = _CAPI_DGLNCCLGetUniqueId(rank)
    id = th.ByteTensor(id)
    th.distributed.broadcast(id, 0)
    _CAPI_DGLNCCLInit(bytearray(id.byte()), rank, world_size)

_init_api("dgl.ds")