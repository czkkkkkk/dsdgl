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

def load_subtensor(train_label, input_nodes, seeds, min_vids, feat_dim):
  input_nodes = F.to_dgl_nd(input_nodes)
  min_vids = F.to_dgl_nd(min_vids)
  ret = _CAPI_DGLDSLoadSubtensor(input_nodes, min_vids)
  ret = F.from_dgl_nd(ret)
  return ret.reshape(-1, feat_dim), train_label[seeds]

_init_api("dgl.ds.load_subtensor")