from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError, EID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils
from ..convert import graph

__all__ = [
    'pin_graph',
    ]

def pin_graph(g, rank):
  g._graph = _CAPI_DGLDSPinGraph(g._graph, rank)
  return g

def test_array(array):
  _CAPI_DGLDSPinGraphTest(array)

_init_api("dgl.ds.pin_graph")