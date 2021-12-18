from dgl._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

@register_object
class DSContext(ObjectBase):
  @staticmethod
  def create():
    return _CAPI_DGLCreateDSContext()

_init_api("dgl.ds")