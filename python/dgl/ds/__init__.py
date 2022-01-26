from .._ffi.function import _init_api

import torch as th

from .sampling import *
from .to_block import to_block
from .pin_graph import pin_graph, test_array
from .load_subtensor import load_subtensor

def init(rank, world_size):
    _CAPI_DGLDSInitialize(rank, world_size)

_init_api("dgl.ds")