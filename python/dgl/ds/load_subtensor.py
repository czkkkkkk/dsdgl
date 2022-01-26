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

def load_subtensor(train_feature, train_label, input_nodes, min_vids):
  train_feature = F.to_dgl_nd(train_feature)
  min_vids = F.to_dgl_nd(min_vids)
  ret = _CAPI_DGLDSLoadSubtensor(train_feature, F.to_dgl_nd(input_nodes), min_vids)
  ret = F.from_dgl_nd(ret)
  return ret.reshape(-1, train_feature.shape[-1]), train_label[input_nodes]

_init_api("dgl.ds.load_subtensor")