import dgl
from .._ffi.function import _init_api
from .. import backend as F


__all__ = [
    'cache_feats',
    ]

def cache_feats(graph, feats, ratio, rank):
    assert ratio >= 0 and ratio <= 100
    if ratio == 100:
        feats = feats.to(rank)
        return feats, None, None
    feats = F.to_dgl_nd(feats)
    global_ids = F.to_dgl_nd(graph.ndata[dgl.NID])
    degs = F.to_dgl_nd(graph.in_degrees())
    ret = _CAPI_DGLDSCacheFeats(feats, global_ids, degs, ratio)
    dev_feats = F.from_dgl_nd(ret(0)).reshape(-1, feats.shape[-1])
    shared_feats = F.from_dgl_nd(ret(1)).reshape(-1, feats.shape[-1])
    feat_pos_map = F.from_dgl_nd(ret(2))
    return dev_feats, shared_feats, feat_pos_map

_init_api("dgl.ds.cache_feats")