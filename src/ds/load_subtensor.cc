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
#include "cuda/cuda_utils.h"

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

  IdArray idx;
  IdArray original_input_nodes = input_nodes.Clone();

  IdArray send_sizes, send_offset;
  std::tie(input_nodes, idx, send_sizes, send_offset) = Partition(input_nodes, min_vids, world_size);
  CUDACHECK(cudaStreamSynchronize(0));
  auto host_send_offset = send_offset.CopyTo(DLContext({kDLCPU, 0}));

  IdArray frontier, host_recv_offset;
  Shuffle(input_nodes, host_send_offset, send_sizes, rank, world_size, context->nccl_comm, &frontier, &host_recv_offset);
  CUDACHECK(cudaStreamSynchronize(0));

  ConvertGidToLid(frontier, min_vids, rank);
  IdArray features_to_send;
  LoadFeature(frontier, features, &features_to_send);
  CUDACHECK(cudaStreamSynchronize(0));

  IdArray features_recv;
  Reshuffle(features_to_send, features->shape[1], n_input_nodes, host_send_offset, host_recv_offset, rank, world_size, context->nccl_comm, &features_recv);
  CUDACHECK(cudaStreamSynchronize(0));
  
  features_recv = Remap(features_recv, idx, features->shape[1]);
  CUDACHECK(cudaStreamSynchronize(0));

  *rv = features_recv;
});

}
}