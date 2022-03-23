from .._ffi.function import _init_api

import torch as th

from .sampling import *
from .pin_graph import pin_graph, test_array
from .cache import cache_feats, cache_graph
from .load_subtensor import load_subtensor
from .. import backend as F

class DummyOp(th.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        import threading
        print('current thread:', threading.get_ident())
        s = th.cuda.current_stream()
        _CAPI_DGLDSSetStream(s._as_parameter_, 0, 2)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        import threading
        print('current thread:', threading.get_ident())
        s = th.cuda.current_stream()
        _CAPI_DGLDSSetStream(s._as_parameter_, 0, 2)
        return grad_output

def init(rank, world_size, thread_num=2, enable_kernel_control=False, enable_comm_control=True):
    _CAPI_DGLDSInitialize(rank, world_size, thread_num, enable_kernel_control, enable_comm_control)

# dgl thread local stream for both forward and backward threads
def set_device_thread_local_stream(device, s):
    with th.cuda.stream(s):
        a = th.full((), 0.0, device=device, dtype=th.float32, requires_grad=True)
        loss = DummyOp.apply(a)
        loss.backward()

# Sampler role = 0, loader role = 1.
def set_thread_local_stream(s, thread_id=0, role=0):
    _CAPI_DGLDSSetStream(s._as_parameter_, thread_id, role)

def set_queue_size(size):
    _CAPI_DGLDSSetQueueSize(size)

def allgather_train_labels(labels):
    labels = F.to_dgl_nd(labels.long())
    labels = _CAPI_DGLDSAllgatherTrainLabels(labels)
    return F.from_dgl_nd(labels)



_init_api("dgl.ds")