import dgl
from .._ffi.function import _init_api
from .. import backend as F


__all__ = [
    'cache_feats',
    'cache_graph'
    ]

def cache_feats(feat_mode, graph, feats, ratio):
    '''
    Return the device features, cached features and feat_pos_map if cache_ratio is smaller than 100
    '''
    assert ratio >= 0 and ratio <= 100
    feats = F.to_dgl_nd(feats)
    global_ids = F.to_dgl_nd(graph.ndata[dgl.NID])
    degs = F.to_dgl_nd(graph.in_degrees())
    _CAPI_DGLDSCacheFeats(feat_mode, feats, global_ids, degs, ratio)

def cache_graph(g, ratio):
    assert ratio >= 0 and ratio <= 100
    # Use out degree because we assume the graph is reversed
    degs = F.to_dgl_nd(g.out_degrees())
    _CAPI_DGLDSCacheGraph(g._graph, ratio, degs)

_init_api("dgl.ds.cache")