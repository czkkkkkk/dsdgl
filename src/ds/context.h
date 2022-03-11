#ifndef DGL_DS_CONTEXT_H_
#define DGL_DS_CONTEXT_H_

#include <nccl.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <dgl/array.h>
#include <memory>
#include <atomic>
#include <dmlc/thread_local.h>

#include "coordinator.h"
#include "./comm/comm_info.h"
#include <atomic>
#include <vector>

using namespace dgl::runtime;

namespace dgl {
namespace ds {

enum FeatMode { kFeatModeAllCache, kFeatModePartitionCache, kFeatModeReplicateCache };

#define ENCODE_SHARED_ID(i) (-(i)-2)

#define SAMPLER_ROLE 0
#define LOADER_ROLE 1

#define THREAD_LOCAL_PINNED_ARRAY_SIZE 256
#define N_PINNED_ARRAY 3

struct DSThreadEntry {
  IdArray pinned_array[N_PINNED_ARRAY];
  int pinned_array_counter;
  static DSThreadEntry* ThreadLocal();
};

struct DSContext {
  bool initialized = false;
  int world_size;
  int rank;
  int thread_num;
  std::vector<ncclComm_t> nccl_comm;
  std::vector<std::unique_ptr<CommInfo> > comm_info;
  std::unique_ptr<Coordinator> coordinator;
  std::unique_ptr<Coordinator> comm_coordinator;

  // Feature related arrays
  bool feat_loaded = false;
  FeatMode feat_mode;
  IdArray dev_feats, shared_feats, feat_pos_map;
  int feat_dim;

  // Kernel controller
  bool enable_kernel_control;
  std::atomic<int> sampler_queue_size{0}, loader_queue_size{0};

  static DSContext* Global() {
    static DSContext instance;
    return &instance;
  }
};

}
}

#endif