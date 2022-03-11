from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError, EID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils
from ..convert import graph, ds_subgraph

__all__ = [
    'sample_neighbors',
    'rebalance_train_nids',
    'sample_neighbors_uva',
    'csr_to_global_id'
    ]

def sample_neighbors(g, num_vertices, device_min_vids, device_min_eids, nodes, fanout, global_nid_map, edge_dir='in', prob=None, replace=True,
                     copy_ndata=True, copy_edata=True, is_local=False):
    prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    device_min_vids = F.to_dgl_nd(device_min_vids)
    device_min_eids = F.to_dgl_nd(device_min_eids)
    nodes = F.to_dgl_nd(nodes)
    global_nid_map = F.to_dgl_nd(global_nid_map)
    subgidx = _CAPI_DGLDSSampleNeighbors(g._graph, num_vertices, device_min_vids, device_min_eids, nodes, 
                                         fanout, edge_dir, prob_arrays, replace, global_nid_map, is_local)

    src, dst, eid = subgidx.edges(0)
    # ret = graph((dst, src), num_nodes = num_vertices)
    # ret = ds_subgraph((dst, src), num_nodes = num_vertices)
    ret = DGLHeteroGraph(subgidx)
    return ret

def rebalance_train_nids(train_nids, batch_size, global_nid_map):
    train_nids = F.to_dgl_nd(train_nids)
    global_nid_map = F.to_dgl_nd(global_nid_map)
    ret = _CAPI_DGLDSRebalanceNIds(train_nids, batch_size, global_nid_map)
    ret = F.from_dgl_nd(ret)
    return ret

def sample_neighbors_uva(g, nodes, num_vertices, fanout, replace=True):
    nodes = F.to_dgl_nd(nodes)
    subgidx = _CAPI_DGLDSSampleNeighborsUVA(g._graph, nodes, fanout, replace)
    src, dst, eid = subgidx.edges(0)
    ret = graph((dst, src), num_nodes = num_vertices)
    return ret

def csr_to_global_id(g, global_nid_map):
    global_nid_map = F.to_dgl_nd(global_nid_map)
    g._graph = _CAPI_DGLDSCSRToGlobalId(g._graph, global_nid_map)
    return g


_init_api("dgl.ds.sampling")