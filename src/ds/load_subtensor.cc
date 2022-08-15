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

NDArray LoadSubtensorsPartitionCacheAllFeats(IdArray input_nodes, IdArray min_vids) {
  auto* context = DSContext::Global();
  int n_input_nodes = input_nodes->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;
  CUDACHECK(cudaSetDevice(rank));
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t s = thr_entry->stream;
  IdArray dev_feats = context->dev_feats;
  int feat_dim = context->feat_dim;

  IdArray send_sizes, send_offset;
  Cluster(rank, input_nodes, min_vids, world_size, &send_sizes, &send_offset);

  IdArray frontier, recv_offset;
  std::tie(frontier, recv_offset) = Alltoall(input_nodes, send_offset, 1, rank, world_size);

  ConvertGidToLid(frontier, min_vids, rank);
  IdArray features_to_send = IdArray::Empty({frontier->shape[0] * feat_dim}, dev_feats->dtype, frontier->ctx);
  IndexSelect(frontier->shape[0], frontier, dev_feats, features_to_send, feat_dim);

  IdArray features_recv, feature_recv_offset;
  std::tie(features_recv, feature_recv_offset) = Alltoall(features_to_send, recv_offset, feat_dim, rank, world_size, send_offset);

  return features_recv;

}

NDArray LoadSubtensorsPartitionCacheSomeFeats(IdArray input_nodes, IdArray min_vids) {
  auto* context = DSContext::Global();
  int n_input_nodes = input_nodes->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;
  CUDACHECK(cudaSetDevice(rank));
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = thr_entry->stream;
  cudaStream_t data_copy_stream = thr_entry->data_copy_stream;
  IdArray dev_feats = context->dev_feats;
  IdArray shared_feats = context->shared_feats;
  IdArray feat_pos_map = context->feat_pos_map;
  int feat_dim = context->feat_dim;
  

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

  if(context->enable_profiler) {
    context->profiler->UpdateFeatCacheRate(n_dev_nodes, n_input_nodes - n_dev_nodes);
  }

  IdArray ret_feats = IdArray::Empty({n_input_nodes * feat_dim}, dev_feats->dtype, dev_feats->ctx);

  // 2. Load device features
  IdArray send_sizes, send_offset;
  Cluster(rank, dev_nodes, min_vids, world_size, &send_sizes, &send_offset);

  IdArray shuffled_dev_nodes, recv_offset;
  std::tie(shuffled_dev_nodes, recv_offset) = Alltoall(dev_nodes, send_offset, 1, rank, world_size);

  IdArray features_to_send = IdArray::Empty({shuffled_dev_nodes->shape[0] * feat_dim}, dev_feats->dtype, shuffled_dev_nodes->ctx);
  IndexSelect(shuffled_dev_nodes->shape[0], shuffled_dev_nodes, dev_feats, features_to_send, feat_dim, feat_pos_map);

  IdArray features_recv, feature_recv_offset;
  std::tie(features_recv, feature_recv_offset) = Alltoall(features_to_send, recv_offset, feat_dim, rank, world_size, send_offset);

  IndexSelect(features_recv->shape[0] / feat_dim, NullArray(), features_recv, ret_feats, feat_dim, NullArray(), dev_index);

  // 3. Load host features in parallel
  IndexSelectUVA(host_sorted_nodes->shape[0], host_sorted_nodes, shared_feats, ret_feats, feat_dim, NullArray(), host_index, data_copy_stream);
  
  CUDACHECK(cudaStreamSynchronize(data_copy_stream));
  return ret_feats;
}

std::pair<std::vector<IdType>, std::vector<IdType>> GatherRemoteNodes(IdArray cpu_host_nodes, IdArray cpu_host_index) {
  auto* context = DSContext::Global();
  std::vector<IdType> remote_nodes, remote_index;
  IdType size = cpu_host_nodes->shape[0];
  auto* cpu_host_nodes_ptr = cpu_host_nodes.Ptr<IdType>();
  auto* cpu_host_index_ptr = cpu_host_index.Ptr<IdType>();
  for(int i = 0; i < size; ++i) {
    auto node = cpu_host_nodes_ptr[i];
    auto idx = cpu_host_index_ptr[i];
    if ((node < context->dist_shared_feat_barrier) != (context->node_rank == 0)) {
      remote_nodes.push_back(node);
      remote_index.push_back(idx);
    }
  }
  return {remote_nodes, remote_index};
}

NDArray GatherRemoteFeats(const std::vector<IdType>& remote_nodes) {
  auto* context = DSContext::Global();
  auto* shared_feat = context->shared_feats.Ptr<DataType>();
  int feat_dim = context->feat_dim;
  auto my_nodes = context->dist_coordinator->PeerExchange(remote_nodes);
  std::vector<DataType> my_feats(my_nodes.size() * feat_dim);
  for(int i = 0; i < my_nodes.size(); ++i) {
    for(int j = 0; j < feat_dim; ++j) {
      my_feats[i * feat_dim + j] = shared_feat[my_nodes[i]*feat_dim+j];
    }
  }
  auto remote_feats = context->dist_coordinator->PeerExchange(my_feats);
  return NDArray::FromVector<DataType>(remote_feats);
}

NDArray LoadSubtensorsDistPartitionCacheSomeFeats(IdArray input_nodes, IdArray min_vids) {
  auto* context = DSContext::Global();
  int n_input_nodes = input_nodes->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;
  CUDACHECK(cudaSetDevice(rank));
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = thr_entry->stream;
  cudaStream_t data_copy_stream = thr_entry->data_copy_stream;
  IdArray dev_feats = context->dev_feats;
  IdArray shared_feats = context->shared_feats;
  IdArray feat_pos_map = context->feat_pos_map;
  int feat_dim = context->feat_dim;
  

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

  if(context->enable_profiler) {
    context->profiler->UpdateFeatCacheRate(n_dev_nodes, n_input_nodes - n_dev_nodes);
  }

  IdArray ret_feats = IdArray::Empty({n_input_nodes * feat_dim}, dev_feats->dtype, dev_feats->ctx);

  // 2. Load device features
  IdArray send_sizes, send_offset;
  Cluster(rank, dev_nodes, min_vids, world_size, &send_sizes, &send_offset);

  IdArray shuffled_dev_nodes, recv_offset;
  std::tie(shuffled_dev_nodes, recv_offset) = Alltoall(dev_nodes, send_offset, 1, rank, world_size);

  IdArray features_to_send = IdArray::Empty({shuffled_dev_nodes->shape[0] * feat_dim}, dev_feats->dtype, shuffled_dev_nodes->ctx);
  IndexSelect(shuffled_dev_nodes->shape[0], shuffled_dev_nodes, dev_feats, features_to_send, feat_dim, feat_pos_map);

  IdArray features_recv, feature_recv_offset;
  std::tie(features_recv, feature_recv_offset) = Alltoall(features_to_send, recv_offset, feat_dim, rank, world_size, send_offset);

  IndexSelect(features_recv->shape[0] / feat_dim, NullArray(), features_recv, ret_feats, feat_dim, NullArray(), dev_index);

  auto cpu_host_nodes = host_sorted_nodes.CopyTo({kDLCPU, 0}, data_copy_stream);
  auto cpu_host_index = host_index.CopyTo({kDLCPU, 0}, data_copy_stream);

  // 3. Load host features in parallel
  IndexSelectUVALocal(host_sorted_nodes->shape[0], host_sorted_nodes, shared_feats, ret_feats, feat_dim, context->dist_shared_feat_barrier, context->node_rank, NullArray(), host_index, stream);

  CUDACHECK(cudaStreamSynchronize(data_copy_stream));

  std::vector<IdType> remote_nodes, remote_index;
  std::tie(remote_nodes, remote_index) = GatherRemoteNodes(cpu_host_nodes, cpu_host_index);
  auto remote_feats = GatherRemoteFeats(remote_nodes);
  remote_feats = remote_feats.CopyTo(input_nodes->ctx, data_copy_stream);
  auto remote_index_array = IdArray::FromVector(remote_index).CopyTo(input_nodes->ctx, data_copy_stream);
  IndexSelect(remote_nodes.size(), NullArray(), remote_feats, ret_feats, context->feat_dim, NullArray(), remote_index_array, data_copy_stream);
  CUDACHECK(cudaStreamSynchronize(data_copy_stream));
  return ret_feats;
}

NDArray LoadSubtensorsReplicateCacheSomeFeats(IdArray input_nodes, IdArray min_vids) {
  auto* context = DSContext::Global();
  int n_input_nodes = input_nodes->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;
  CUDACHECK(cudaSetDevice(rank));
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = thr_entry->stream;
  cudaStream_t data_copy_stream = thr_entry->data_copy_stream;
  IdArray dev_feats = context->dev_feats;
  IdArray shared_feats = context->shared_feats;
  IdArray feat_pos_map = context->feat_pos_map;
  int feat_dim = context->feat_dim;

  // 1. Divide the input nodes into two subsets: on/off device
  IdArray part_ids, part_nodes, part_offset;
  std::tie(part_ids, part_nodes, part_offset) = GetFeatTypePartIds(input_nodes, feat_pos_map);
  CUDACHECK(cudaStreamSynchronize(stream));
  IdArray host_part_offset = part_offset.CopyTo({kDLCPU, 0}, data_copy_stream);
  IdArray sorted_nodes, index;
  std::tie(sorted_nodes, index) = MultiWayScan(part_nodes, part_offset, part_ids, 2);
  CUDACHECK(cudaStreamSynchronize(data_copy_stream));
  IdType n_dev_nodes = host_part_offset.Ptr<IdType>()[1];
  // dev_nodes are indices of cached features
  IdArray dev_nodes = sorted_nodes.CreateView({n_dev_nodes}, sorted_nodes->dtype, 0);
  IdArray dev_index = index.CreateView({n_dev_nodes}, index->dtype, 0);

  // host nodes are shared feature indices
  IdArray host_sorted_nodes = sorted_nodes.CreateView({n_input_nodes - n_dev_nodes}, sorted_nodes->dtype, n_dev_nodes * sorted_nodes->dtype.bits / 8);
  IdArray host_index = index.CreateView({n_input_nodes - n_dev_nodes}, index->dtype, n_dev_nodes * index->dtype.bits / 8);

  if(context->enable_profiler) {
    context->profiler->UpdateFeatCacheRate(n_dev_nodes, n_input_nodes - n_dev_nodes);
  }

  IdArray ret_feats = IdArray::Empty({n_input_nodes * feat_dim}, dev_feats->dtype, dev_feats->ctx);

  IndexSelect(dev_nodes->shape[0], dev_nodes, dev_feats, ret_feats, feat_dim, feat_pos_map);

  IndexSelect(host_sorted_nodes->shape[0], host_sorted_nodes, shared_feats, ret_feats, feat_dim, NullArray(), host_index, data_copy_stream);
  CUDACHECK(cudaStreamSynchronize(data_copy_stream));
  return ret_feats;
}

DGL_REGISTER_GLOBAL("ds.load_subtensor._CAPI_DGLDSLoadSubtensor")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  IdArray input_nodes = args[0];
  IdArray min_vids = args[1];


  auto* context = DSContext::Global();
  CHECK(context->feat_loaded);

  if(context->feat_mode == kFeatModeAllCache) {
    *rv = LoadSubtensorsPartitionCacheAllFeats(input_nodes, min_vids);
  } else if(context->feat_mode == kFeatModePartitionCache) {
    *rv = LoadSubtensorsPartitionCacheSomeFeats(input_nodes, min_vids);
  } else if(context->feat_mode == kFeatModeDistPartitionCache) {
    *rv = LoadSubtensorsDistPartitionCacheSomeFeats(input_nodes, min_vids);
  } else if (context->feat_mode == kFeatModeReplicateCache) {
    *rv = LoadSubtensorsReplicateCacheSomeFeats(input_nodes, min_vids);
  } else {
    CHECK(false);
  }
  CUDACHECK(cudaStreamSynchronize(CUDAThreadEntry::ThreadLocal()->stream));
});


}
}
