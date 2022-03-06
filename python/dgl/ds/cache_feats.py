import dgl
from .._ffi.function import _init_api
from .. import backend as F


__all__ = [
    'cache_feats',
    ]

def cache_feats(feat_mode, graph, feats, ratio):
    '''
    Return the device features, cached features and feat_pos_map if cache_ratio is smaller than 100
    '''
    assert ratio >= 0 and ratio <= 100
    # feat_mode = feat_mode.encode('utf-8')
    feats = F.to_dgl_nd(feats)
    global_ids = F.to_dgl_nd(graph.ndata[dgl.NID])
    degs = F.to_dgl_nd(graph.in_degrees())
    _CAPI_DGLDSCacheFeats(feat_mode, feats, global_ids, degs, ratio)

_init_api("dgl.ds.cache_feats")