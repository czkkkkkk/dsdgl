from dgl._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

@register_object
class DSContext(ObjectBase):
  @staticmethod
  def create(rank):
    return _CAPI_DGLCreateDSContext(rank)

_init_api("dgl.ds.context")