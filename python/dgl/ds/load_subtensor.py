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

def load_subtensor(train_feature, input_nodes, min_vids):
  train_feature = F.to_dgl_nd(train_feature)
  input_nodes = F.to_dgl_nd(input_nodes)
  min_vids = F.to_dgl_nd(min_vids)
  ret = _CAPI_DGLDSLoadSubtensor(train_feature, input_nodes, min_vids)
  #ret = F.tensor(ret, dtype=F.float)
  #return ret.reshape(input_nodes.shape[0], train_feature.shape[1])
  return ret

_init_api("dgl.ds.load_subtensor")