#include <nccl.h>
#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/sampling/neighbor.h>
#include <cuda_runtime.h>
#include <dgl/runtime/device_api.h>
#include <stdio.h>

#include "../c_api_common.h"
#include "../graph/unit_graph.h"
#include "context.h"
#include "cuda/ds_kernel.h"
#include "cuda/alltoall.h"
#include "cuda/cuda_utils.h"
#include <chrono>
#include <thread>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace ds {

DGL_REGISTER_GLOBAL("ds.load_subtensor._CAPI_DGLDSLoadSubtensor")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  IdArray features = args[0];
  IdArray input_nodes = args[1];
  IdArray min_vids = args[2];

  auto* context = DSContext::Global();
  int n_input_nodes = input_nodes->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;
  CUDACHECK(cudaSetDevice(rank));
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t s = thr_entry->stream;

  IdArray send_sizes, send_offset;
  Cluster(rank, input_nodes, min_vids, world_size, &send_sizes, &send_offset);

  IdArray frontier, recv_offset;
  std::tie(frontier, recv_offset) = Alltoall(input_nodes, send_offset, 1, rank, world_size);

  ConvertGidToLid(frontier, min_vids, rank);
  IdArray features_to_send;
  LoadFeature(frontier, features, &features_to_send);

  IdArray features_recv, feature_recv_offset;
  std::tie(features_recv, feature_recv_offset) = Alltoall(features_to_send, recv_offset, features->shape[1], rank, world_size);
  
  features_recv = Remap(features_recv, idx, features->shape[1]);

  *rv = features_recv;
});

}
}
