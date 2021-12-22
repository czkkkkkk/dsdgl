from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError, EID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils

__all__ = [
    'sample_neighbors',
    ]

# def sample_neighbors(g, gpb, seeds, fanout, context):
#     frontier = _CAPI_DGLDSSampleNeighbors(g, seeds, fanout, context)
#     return frontier

# def sample_neighbors(g, device_min_vids, nodes, fanout, context, edge_dir='in', prob=None, replace=True,
#                      copy_ndata=True, copy_edata=True):
#     if not isinstance(nodes, dict):
#         assert(len(g.ntypes) == 1)
#         device_min_vids = {g.ntypes[0] : device_min_vids}
#         nodes = {g.ntypes[0] : nodes}
#     device_min_vids = utils.prepare_tensor_dict(g, device_min_vids, 'min_vids')
#     nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
#     device_min_vids_all_types = []
#     nodes_all_types = []
#     for ntype in g.ntypes:
#         device_min_vids_all_types.append(F.to_dgl_nd(device_min_vids[ntype]))
#         nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))

#     fanout_array = fanout
#     prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
#     subgidx = _CAPI_DGLDSSampleNeighbors(g._graph, device_min_vids_all_types, nodes_all_types, 
#                                          fanout_array, edge_dir, prob_arrays, replace, context)
#     induced_edges = subgidx.induced_edges
#     ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)

#     # todo zqh handle features
#     # if copy_ndata:
#     #     node_frames = utils.extract_node_subframes(g, None)
#     #     utils.set_new_frames(ret, node_frames=node_frames)
#     # if copy_edata:
#     #     edge_frames = utils.extract_edge_subframes(g, induced_edges)
#     #     utils.set_new_frames(ret, edge_frames=edge_frames)
#     return ret

def sample_neighbors(g, num_vertices, device_min_vids, device_min_eids, nodes, fanout, context, edge_dir='in', prob=None, replace=True,
                     copy_ndata=True, copy_edata=True):
    prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    print("graph:", g._graph)
    print("device_min_vids:", device_min_vids)
    print("device_min_eids:", device_min_eids)
    print("nodes:", nodes)
    print("fanout:", fanout)
    print("context:", context)
    print("edge_dir:", edge_dir)
    print("prob_arrays:", prob_arrays)
    print("replace:", replace)
    device_min_vids = F.to_dgl_nd(device_min_vids)
    device_min_eids = F.to_dgl_nd(device_min_eids)
    nodes = F.to_dgl_nd(nodes)
    subgidx = _CAPI_DGLDSSampleNeighbors(g._graph, num_vertices, device_min_vids, device_min_eids, nodes, 
                                         fanout, edge_dir, prob_arrays, replace, context)
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)

    # todo zqh handle features
    # if copy_ndata:
    #     node_frames = utils.extract_node_subframes(g, None)
    #     utils.set_new_frames(ret, node_frames=node_frames)
    # if copy_edata:
    #     edge_frames = utils.extract_edge_subframes(g, induced_edges)
    #     utils.set_new_frames(ret, edge_frames=edge_frames)
    return ret

_init_api("dgl.ds.sampling")