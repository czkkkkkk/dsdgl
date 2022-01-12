from .._ffi.function import _init_api
from .. import backend as F
from .. import utils
from ..heterograph import DGLHeteroGraph, DGLBlock
from collections.abc import Iterable, Mapping

__all__ = [
    'to_block',
    ]

def to_block(g, dst_nodes=None, include_dst_in_src=True):
  if dst_nodes is None:
    # Find all nodes that appeared as destinations
    dst_nodes = defaultdict(list)
    for etype in g.canonical_etypes:
      _, dst = g.edges(etype=etype)
      dst_nodes[etype[2]].append(dst)
    dst_nodes = {ntype: F.unique(F.cat(values, 0)) for ntype, values in dst_nodes.items()}
  elif not isinstance(dst_nodes, Mapping):
    # dst_nodes is a Tensor, check if the g has only one type.
    if len(g.ntypes) > 1:
      raise DGLError(
        'Graph has more than one node type; please specify a dict for dst_nodes.')
    dst_nodes = {g.ntypes[0]: dst_nodes}

  dst_node_ids = [
    utils.toindex(dst_nodes.get(ntype, []), g._idtype_str).tousertensor(
      ctx=F.to_backend_ctx(g._graph.ctx))
    for ntype in g.ntypes]
  dst_node_ids_nd = [F.to_dgl_nd(nodes) for nodes in dst_node_ids]

  for d in dst_node_ids_nd:
    if g._graph.ctx != d.ctx:
      raise ValueError('g and dst_nodes need to have the same context.')

  new_graph_index, src_nodes_nd, induced_edges_nd = _CAPI_DGLDSToBlock(
    g._graph, dst_node_ids_nd, include_dst_in_src)

  # The new graph duplicates the original node types to SRC and DST sets.
  new_ntypes = (g.ntypes, g.ntypes)
  new_graph = DGLBlock(new_graph_index, new_ntypes, g.etypes)
  assert new_graph.is_unibipartite  # sanity check

  src_node_ids = [F.from_dgl_nd(src) for src in src_nodes_nd]
  edge_ids = [F.from_dgl_nd(eid) for eid in induced_edges_nd]

  node_frames = utils.extract_node_subframes_for_block(g, src_node_ids, dst_node_ids)
  edge_frames = utils.extract_edge_subframes(g, edge_ids)
  utils.set_new_frames(new_graph, node_frames=node_frames, edge_frames=edge_frames)

  return new_graph

_init_api("dgl.ds.to_block")