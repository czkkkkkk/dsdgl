#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <algorithm>
#include <numeric>

#include "../c_api_common.h"
#include "./cuda/ds_kernel.h"
#include "./context.h"

using namespace dgl::runtime; 

namespace dgl {
namespace ds {

template<typename T>
std::vector<T> Flatten(const std::vector<std::vector<T>>& input) {
  std::vector<T> output;
  for(const auto& vec: input) {
    for(auto v: vec) {
      output.push_back(v);
    }
  }
  return output;
}

/**
 * @brief Partition features into two subsets: CPU features and GPU features, where all GPUs shared the same CPU features and each GPU have their own subset of features. The function returns device/shared features and feature position map. 
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
 * 
 * @return (dev_feats, shared_feats, FPM)
 * 
 */
DGL_REGISTER_GLOBAL("ds.cache_feats._CAPI_DGLDSCacheFeats")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  IdArray feats = args[0];
  IdArray global_ids = args[1];
  IdArray local_degrees = args[2];
  int ratio = args[3];

  int n_local_nodes = feats->shape[0];
  int feat_dim = feats->shape[1];
  // TODO set the ratio
  int n_dev_nodes = n_local_nodes * (ratio / 100.);
  int rank = DSContext::Global()->rank;
  auto* coor = DSContext::Global()->coordinator.get();

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
  auto gathered_feats = coor->Gather(local_shared_feats);
  IdArray shared_feats = NullArray(feats->dtype, feats->ctx);
  std::vector<IdType> feat_pos_map;
  IdType n_shared_nodes = 0;
  if(coor->IsRoot()) {
    IdType n_nodes = 0;
    for(auto c: gathered_n_nodes) {
      n_nodes += c;
    }
    auto flatten_ids = Flatten(gathered_ids);
    n_shared_nodes = flatten_ids.size();
    auto flatten_feats = Flatten(gathered_feats);
    feat_pos_map.resize(n_nodes, -1);
    for(int i = 0; i < flatten_ids.size(); ++i) {
      IdType global_nid = flatten_ids[i];
      feat_pos_map[global_nid] = - i - 2;
    }
    shared_feats = IdArray::FromVector<DataType>(flatten_feats);
  }
  shared_feats = CreateShmArray(shared_feats, "shared_features");
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
  IdArray dev_feats = IdArray::FromVector(local_dev_feats, {kDLGPU, rank});
  IdArray feat_pos_map_array = IdArray::FromVector(feat_pos_map, {kDLGPU, rank});
  *rv = ConvertNDArrayVectorToPackedFunc({dev_feats, shared_feats, feat_pos_map_array});
});

}
}