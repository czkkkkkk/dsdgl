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
#include "cuda/scan.h"
#include <chrono>
#include <thread>

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
  int feat_dim = features->shape[1];

  IdArray send_sizes, send_offset;
  Cluster(rank, input_nodes, min_vids, world_size, &send_sizes, &send_offset);

  IdArray frontier, recv_offset;
  std::tie(frontier, recv_offset) = Alltoall(input_nodes, send_offset, 1, rank, world_size);

  ConvertGidToLid(frontier, min_vids, rank);
  IdArray features_to_send = IdArray::Empty({frontier->shape[0] * features->shape[1]}, features->dtype, frontier->ctx);
  IndexSelect(frontier->shape[0], frontier, features, features_to_send, feat_dim);

  IdArray features_recv, feature_recv_offset;
  std::tie(features_recv, feature_recv_offset) = Alltoall(features_to_send, recv_offset, features->shape[1], rank, world_size);

  *rv = features_recv;
});


DGL_REGISTER_GLOBAL("ds.load_subtensor._CAPI_DGLDSLoadSubtensorWithSharedFeats")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  IdArray dev_feats = args[0];
  IdArray shared_feats = args[1];
  IdArray feat_pos_map = args[2];
  IdArray input_nodes = args[3];
  IdArray min_vids = args[4];

  auto* context = DSContext::Global();
  int n_input_nodes = input_nodes->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;
  CUDACHECK(cudaSetDevice(rank));
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = thr_entry->stream;
  cudaStream_t data_copy_stream = thr_entry->data_copy_stream;
  int feat_dim = dev_feats->shape[1];
  

  // 1. Divide the input nodes into two subsets: on/off device
  IdArray part_ids, part_nodes, part_offset;
  std::tie(part_ids, part_nodes, part_offset) = GetFeatTypePartIds(input_nodes, feat_pos_map);
  CUDACHECK(cudaStreamSynchronize(stream));
  IdArray host_part_offset = part_offset.CopyTo({kDLCPU, 0}, data_copy_stream);
  IdArray sorted_nodes, index;
  std::tie(sorted_nodes, index) = MultiWayScan(part_nodes, part_offset, part_ids, 2);
  CUDACHECK(cudaStreamSynchronize(data_copy_stream));
  IdType n_dev_nodes = host_part_offset.Ptr<IdType>()[1];
  // dev_nodes are global nids
  IdArray dev_nodes = sorted_nodes.CreateView({n_dev_nodes}, sorted_nodes->dtype, 0);
  IdArray dev_index = index.CreateView({n_dev_nodes}, index->dtype, 0);

  // host nodes are shared feature indices
  IdArray host_sorted_nodes = sorted_nodes.CreateView({n_input_nodes - n_dev_nodes}, sorted_nodes->dtype, n_dev_nodes * sorted_nodes->dtype.bits / 8);
  IdArray host_index = index.CreateView({n_input_nodes - n_dev_nodes}, index->dtype, n_dev_nodes * index->dtype.bits / 8);

  IdArray ret_feats = IdArray::Empty({n_input_nodes * feat_dim}, dev_feats->dtype, dev_feats->ctx);

  // 2. Load device features
  IdArray send_sizes, send_offset;
  Cluster(rank, dev_nodes, min_vids, world_size, &send_sizes, &send_offset);

  IdArray shuffled_dev_nodes, recv_offset;
  std::tie(shuffled_dev_nodes, recv_offset) = Alltoall(dev_nodes, send_offset, 1, rank, world_size, context->nccl_comm_load, false);

  IdArray features_to_send = IdArray::Empty({shuffled_dev_nodes->shape[0] * feat_dim}, dev_feats->dtype, shuffled_dev_nodes->ctx);
  IndexSelect(shuffled_dev_nodes->shape[0], shuffled_dev_nodes, dev_feats, features_to_send, feat_dim, feat_pos_map);

  IdArray features_recv, feature_recv_offset;
  std::tie(features_recv, feature_recv_offset) = Alltoall(features_to_send, recv_offset, feat_dim, rank, world_size, context->nccl_comm_load, false);

  IndexSelect(features_recv->shape[0] / feat_dim, NullArray(), features_recv, ret_feats, feat_dim, NullArray(), dev_index);

  // 3. Load host features in parallel
  IndexSelect(host_sorted_nodes->shape[0], host_sorted_nodes, shared_feats, ret_feats, feat_dim, NullArray(), host_index, data_copy_stream);
  CUDACHECK(cudaStreamSynchronize(data_copy_stream));
  *rv = ret_feats;
});

}
}
