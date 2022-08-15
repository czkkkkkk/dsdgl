#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <algorithm>
#include <numeric>

#include "../c_api_common.h"
#include "./cuda/ds_kernel.h"
#include "./context.h"
#include "./utils.h"

using namespace dgl::runtime; 

namespace dgl {
namespace ds {


void PartitionCacheAllFeats(IdArray feats) {
  auto* context = DSContext::Global();
  context->feat_mode = kFeatModeAllCache;
  context->feat_dim = feats->shape[1];
  context->dev_feats = feats.CopyTo({kDLGPU, context->rank});
  context->feat_loaded = true;
}

/**
 * @brief Partition features into two subsets: CPU features and GPU features, where all GPUs shared the same CPU features and each GPU have their own subset of features. This function caches device/shared features and feature position map in the DSContext.
 * 
 * Rules for the feature position map (FPM).
 * 1. Each GPU has the same size of FPM, where the size is equal to the global nodes.
 * 2. If the feature of a global node `v` is on the current GPU, then `FPM[v]=i`, where `i` is the index of this node in the `dev_feats`.
 * 3. If the feature of a global node `v` is on CPU, then `FPM[v] = - i - 2`, where `i` is the index of this node in the `shared_feats`.
 * 4. Finally, if the feature of a global `v` is on the other GPU, `FPM[v] = -1`.
 * 
 * We use FPM to lookup features when loading subtensor.
 * 
 * @param feats Partitioned features on CPU
 * @param global_ids Global id map
 * @param local_degrees Degress of each node in the graph
 * @param ratio Cache ratio
 * @return
 */
void PartitionCacheSomeFeats(IdArray feats, IdArray global_ids, IdArray local_degrees, double ratio) {
  int n_local_nodes = feats->shape[0];
  int feat_dim = feats->shape[1];
  int n_dev_nodes = n_local_nodes * (ratio / 100.);
  auto* ds_ctx = DSContext::Global();
  int rank = ds_ctx->rank;
  auto* coor = ds_ctx->coordinator.get();

  // Sort the local nodes according to their degree
  // Partition local nodes into local nodes and shared nodes
  // Send shared nodes and shared feats to the root
  // Root creates a shared memory for collected shared feats and sent the feat_pos_map to all ranks
  // Each rank receives the feat_pos_map and maps the feat_pos_map to the local dev_feats
  std::vector<IdType> sorted_local_ids(n_local_nodes);
  std::iota(sorted_local_ids.begin(), sorted_local_ids.end(), 0);
  std::sort(sorted_local_ids.begin(), sorted_local_ids.end(), [&](IdType l, IdType r) {
    auto* deg_ptr = local_degrees.Ptr<IdType>();
    return deg_ptr[l] > deg_ptr[r];
  });

  std::vector<IdType> local_shared_ids;
  std::vector<DataType> local_shared_feats;
  for(int i = n_dev_nodes; i < n_local_nodes; ++i) {
    IdType idx = sorted_local_ids[i];
    local_shared_ids.push_back(global_ids.Ptr<IdType>()[idx]);
    for(int j = 0; j < feat_dim; ++j) {
      local_shared_feats.push_back(feats.Ptr<DataType>()[idx*feat_dim+j]);
    }
  }
  auto gathered_n_nodes = coor->Gather(n_local_nodes);
  auto gathered_ids = coor->Gather(local_shared_ids);
  auto gathered_feats = coor->GatherLargeVector(local_shared_feats);
  IdArray shared_feats = NullArray(feats->dtype, feats->ctx);
  std::vector<IdType> feat_pos_map;
  if(coor->IsRoot()) {
    IdType n_nodes = 0, n_shared_nodes = 0;
    for(auto c: gathered_n_nodes) {
      n_nodes += c;
    }
    auto flatten_ids = Flatten(gathered_ids);
    n_shared_nodes = flatten_ids.size();

    // auto flatten_feats = std::vector<DataType>(n_shared_nodes * feat_dim, 1);
    auto flatten_feats = Flatten(gathered_feats);
    feat_pos_map.resize(n_nodes, -1);
    for(int i = 0; i < flatten_ids.size(); ++i) {
      IdType global_nid = flatten_ids[i];
      feat_pos_map[global_nid] = ENCODE_ID(i);
    }
    shared_feats = IdArray::FromVector<DataType>(flatten_feats);
  }
  shared_feats = CreateShmArray(shared_feats, "dsdgl_partition_cache_host_feats");
  // Get the global feat_pos_map
  coor->Broadcast(feat_pos_map);
  std::vector<DataType> local_dev_feats;
  for(int i = 0; i < n_dev_nodes; ++i) {
    auto* global_ids_ptr = global_ids.Ptr<IdType>();
    IdType idx = sorted_local_ids[i];
    IdType global_id = global_ids_ptr[idx];
    CHECK(global_id >= 0 && global_id < feat_pos_map.size());
    feat_pos_map[global_id] = i;
    for(int j = 0; j < feat_dim; ++j) {
      local_dev_feats.push_back(feats.Ptr<DataType>()[idx*feat_dim+j]);
    }
  }
  ds_ctx->feat_mode = kFeatModePartitionCache;
  ds_ctx->feat_loaded = true;
  ds_ctx->feat_dim = feat_dim;
  ds_ctx->dev_feats = IdArray::FromVector(local_dev_feats, {kDLGPU, rank});
  ds_ctx->shared_feats = shared_feats;
  ds_ctx->feat_pos_map = IdArray::FromVector(feat_pos_map, {kDLGPU, rank});
}

void DistPartitionCacheSomeFeats(IdArray feats, IdArray global_ids, IdArray local_degrees, double ratio) {
  int n_local_nodes = feats->shape[0];
  int feat_dim = feats->shape[1];
  int n_dev_nodes = n_local_nodes * (ratio / 100.);
  auto* ds_ctx = DSContext::Global();
  int rank = ds_ctx->rank;
  auto* local_coor = ds_ctx->local_coordinator.get();

  // Sort the local nodes according to their degree
  // Partition local nodes into local nodes and shared nodes
  // Send shared nodes and shared feats to the root
  // Root creates a shared memory for collected shared feats and sent the feat_pos_map to all ranks
  // Each rank receives the feat_pos_map and maps the feat_pos_map to the local dev_feats
  std::vector<IdType> sorted_local_ids(n_local_nodes);
  std::iota(sorted_local_ids.begin(), sorted_local_ids.end(), 0);
  std::sort(sorted_local_ids.begin(), sorted_local_ids.end(), [&](IdType l, IdType r) {
    auto* deg_ptr = local_degrees.Ptr<IdType>();
    return deg_ptr[l] > deg_ptr[r];
  });

  std::vector<IdType> local_shared_ids;
  std::vector<DataType> local_shared_feats;
  for(int i = n_dev_nodes; i < n_local_nodes; ++i) {
    IdType idx = sorted_local_ids[i];
    local_shared_ids.push_back(global_ids.Ptr<IdType>()[idx]);
    for(int j = 0; j < feat_dim; ++j) {
      local_shared_feats.push_back(feats.Ptr<DataType>()[idx*feat_dim+j]);
    }
  }
  auto gathered_n_nodes = local_coor->Gather(n_local_nodes);
  auto gathered_ids = local_coor->Gather(local_shared_ids);
  auto gathered_feats = local_coor->GatherLargeVector(local_shared_feats);
  IdArray shared_feats = NullArray(feats->dtype, feats->ctx);
  std::vector<IdType> feat_pos_map;
  IdType n_shared_nodes = 0;
  if(local_coor->IsRoot()) {
    IdType n_nodes = 0;
    for(auto c: gathered_n_nodes) {
      n_nodes += c;
    }
    auto flatten_ids = Flatten(gathered_ids);
    n_shared_nodes = flatten_ids.size();

    // auto flatten_feats = std::vector<DataType>(n_shared_nodes * feat_dim, 1);
    auto flatten_feats = Flatten(gathered_feats);
    feat_pos_map.resize(n_nodes, -1);
    for(int i = 0; i < flatten_ids.size(); ++i) {
      IdType global_nid = flatten_ids[i];
      feat_pos_map[global_nid] = ENCODE_ID(i);
    }
    shared_feats = IdArray::FromVector<DataType>(flatten_feats);
  }
  LOG(ERROR) << "Before create shm";
  shared_feats = CreateShmArray(shared_feats, "dsdgl_partition_cache_host_feats");
  LOG(ERROR) << "After create shm";
  // Get the global feat_pos_map
  local_coor->Broadcast(feat_pos_map);
  local_coor->Broadcast(n_shared_nodes);
  std::vector<DataType> local_dev_feats;
  for(int i = 0; i < n_dev_nodes; ++i) {
    auto* global_ids_ptr = global_ids.Ptr<IdType>();
    IdType idx = sorted_local_ids[i];
    IdType global_id = global_ids_ptr[idx];
    CHECK(global_id >= 0 && global_id < feat_pos_map.size());
    feat_pos_map[global_id] = i;
    for(int j = 0; j < feat_dim; ++j) {
      local_dev_feats.push_back(feats.Ptr<DataType>()[idx*feat_dim+j]);
    }
  }
  ds_ctx->feat_mode = kFeatModeDistPartitionCache;
  ds_ctx->feat_loaded = true;
  ds_ctx->feat_dim = feat_dim;
  ds_ctx->dev_feats = IdArray::FromVector(local_dev_feats, {kDLGPU, rank});
  ds_ctx->shared_feats = shared_feats;
  ds_ctx->feat_pos_map = IdArray::FromVector(feat_pos_map, {kDLGPU, rank});
  ds_ctx->dist_shared_feat_barrier = n_shared_nodes / 2;
  LOG(ERROR) << "Finished cache feats";
}

/**
 * @brief Partition features into two subsets: CPU features and GPU features, where all GPUs shared the same CPU features and GPU features replicate among all GPUs. This function caches device/shared features and feature position map in the DSContext.
 * 
 * Rules for the feature position map (FPM).
 * 1. Each GPU has the same size of FPM, where the size is equal to the global nodes.
 * 2. If the feature of a global node `v` is on the GPU, then `FPM[v]=i`, where `i` is the index of this node in the `dev_feats`.
 * 3. If the feature of a global node `v` is on CPU, then `FPM[v] = - i - 2`, where `i` is the index of this node in the `shared_feats`.
 * 
 * We use FPM to lookup features when loading subtensor.
 * 
 * @param feats Partitioned features on CPU
 * @param global_ids Global id map
 * @param local_degrees Degress of each node in the graph
 * @param ratio Cache ratio
 * @return
 */
void ReplicateCacheSomeFeats(IdArray feats, IdArray global_ids, IdArray local_degrees, double ratio) {
  int n_local_nodes = feats->shape[0];
  int feat_dim = feats->shape[1];
  auto* ds_ctx = DSContext::Global();
  int rank = ds_ctx->rank;
  auto* coor = ds_ctx->coordinator.get();
  // Gather global ids and feats
  std::vector<IdType> global_ids_vec = global_ids.CreateView({n_local_nodes}, global_ids->dtype).ToVector<IdType>();
  // Create 1d arrary view
  std::vector<DataType> feats_vec = feats.CreateView({n_local_nodes * feat_dim}, feats->dtype).ToVector<DataType>();
  std::vector<IdType> degree_vec = local_degrees.CreateView({n_local_nodes}, local_degrees->dtype).ToVector<IdType>();
  auto gathered_global_ids = coor->Gather(global_ids_vec);
  auto gathered_feats = coor->Gather(feats_vec);
  auto gathered_degs = coor->Gather(degree_vec);
  std::vector<IdType> feat_pos_map;
  std::vector<DataType> dev_feats_vec, shared_feats_vec;
  IdArray shared_feats = NullArray({feats->dtype});
  if(coor->IsRoot()) {
    auto flatten_global_ids = Flatten(gathered_global_ids);
    auto flatten_feats = Flatten(gathered_feats);
    auto flatten_degs = Flatten(gathered_degs);
    IdType n_nodes = flatten_global_ids.size();
    std::vector<IdType> index(n_nodes);
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [&](size_t l, size_t r){
      return flatten_degs[l] > flatten_degs[r];
    });
    int n_dev_nodes, n_shared_nodes;
    n_dev_nodes = n_nodes * (ratio / 100.);
    n_shared_nodes = n_nodes - n_dev_nodes;
    feat_pos_map.resize(n_nodes, -1);
    for(int i = 0; i < index.size(); ++i) {
      IdType idx = index[i];
      IdType gid = flatten_global_ids[idx];
      if(i < n_dev_nodes) {
        feat_pos_map[gid] = i;
        for(int j = 0; j < feat_dim; ++j) {
          dev_feats_vec.push_back(flatten_feats[idx*feat_dim+j]);
        }
      } else {
        feat_pos_map[gid] = ENCODE_ID(i - n_dev_nodes);
        for(int j = 0; j < feat_dim; ++j) {
          shared_feats_vec.push_back(flatten_feats[idx*feat_dim+j]);
        }
      }
    }
    for(auto p: feat_pos_map) {
      CHECK_NE(p, -1);
    }
    shared_feats = IdArray::FromVector(shared_feats_vec);
  }
  coor->Broadcast(feat_pos_map);
  coor->Broadcast(dev_feats_vec);

  ds_ctx->feat_mode = kFeatModeReplicateCache;
  ds_ctx->feat_loaded = true;
  ds_ctx->feat_dim = feat_dim;
  ds_ctx->dev_feats = IdArray::FromVector(dev_feats_vec, {kDLGPU, rank});
  ds_ctx->shared_feats = CreateShmArray(shared_feats, "dsdgl_replicate_cache_host_feats");
  ds_ctx->feat_pos_map = IdArray::FromVector(feat_pos_map, {kDLGPU, rank});
}

DGL_REGISTER_GLOBAL("ds.cache._CAPI_DGLDSCacheFeats")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  std::string feat_mode = args[0];
  IdArray feats = args[1];
  LOG(INFO) << "Feature cache mode: " << feat_mode;
  if(feat_mode == "AllCache") {
    PartitionCacheAllFeats(feats);
  } else if(feat_mode == "PartitionCache") {
    IdArray global_ids = args[2];
    IdArray local_degrees = args[3];
    double ratio = args[4];
    LOG(INFO) << "Feature cache ratio: " << ratio;
    CHECK(global_ids->dtype.bits == 64);
    CHECK(local_degrees->dtype.bits == 64);
    PartitionCacheSomeFeats(feats, global_ids, local_degrees, ratio);
  } else if(feat_mode == "DistPartitionCache") {
    IdArray global_ids = args[2];
    IdArray local_degrees = args[3];
    double ratio = args[4];
    LOG(INFO) << "Feature cache ratio: " << ratio;
    CHECK(global_ids->dtype.bits == 64);
    CHECK(local_degrees->dtype.bits == 64);
    DistPartitionCacheSomeFeats(feats, global_ids, local_degrees, ratio);
  } else if(feat_mode == "ReplicateCache") {
    IdArray global_ids = args[2];
    IdArray local_degrees = args[3];
    double ratio = args[4];
    LOG(INFO) << "Feature cache ratio: " << ratio;
    CHECK(global_ids->dtype.bits == 64);
    CHECK(local_degrees->dtype.bits == 64);
    ReplicateCacheSomeFeats(feats, global_ids, local_degrees, ratio);
  } else {
    LOG(FATAL) << "Unsupported feature cache mode: " << feat_mode;
  }

});

}
}