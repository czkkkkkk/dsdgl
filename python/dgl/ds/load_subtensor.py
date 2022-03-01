from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError, EID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils
from ..convert import graph

__all__ = [
    'load_subtensor',
    ]

def load_subtensor(dev_feats, train_label, input_nodes, seeds, min_vids, shared_feats=None, feat_pos_map=None):
  dev_feats = F.to_dgl_nd(dev_feats)
  min_vids = F.to_dgl_nd(min_vids)
  if shared_feats is None:
    ret = _CAPI_DGLDSLoadSubtensor(dev_feats, F.to_dgl_nd(input_nodes), min_vids)
  else:
    shared_feats = F.to_dgl_nd(shared_feats)
    feat_pos_map = F.to_dgl_nd(feat_pos_map)
    ret = _CAPI_DGLDSLoadSubtensorWithSharedFeats(dev_feats, shared_feats, feat_pos_map, F.to_dgl_nd(input_nodes), min_vids)
  ret = F.from_dgl_nd(ret)
  return ret.reshape(-1, dev_feats.shape[-1]), train_label[seeds]

_init_api("dgl.ds.load_subtensor")