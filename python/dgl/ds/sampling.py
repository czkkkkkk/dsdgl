from .._ffi.function import _init_api


def sample_neighbors(g, seeds, fanout, context):
    frontier = _CAPI_DGLDSSampleNeighbors(g, seeds, fanout, context)
    return frontier

_init_api("dgl.ds")