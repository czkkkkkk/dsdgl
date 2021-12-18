from .._ffi.function import _init_api

import torch as th

from .context import DSContext



DS_CONTEXT = None

def init(rank, world_size):
    global DS_CONTEXT
    DS_CONTEXT = DSContext.create()

    id = _CAPI_DGLNCCLGetUniqueId(rank)
    id = th.ByteTensor(id)
    th.distributed.broadcast(id, 0)
    _CAPI_DGLNCCLInit(bytearray(id.byte()), rank, world_size, DS_CONTEXT)


_init_api("dgl.ds")