#ifndef DGL_DS_CONTEXT_H_
#define DGL_DS_CONTEXT_H_

#include <nccl.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <memory>

#include "coordinator.h"
#include "./comm/comm_info.h"
#include <atomic>

using namespace dgl::runtime;

namespace dgl {
namespace ds {

struct DSContext {
  bool initialized = false;
  int world_size;
  int rank;
  std::unique_ptr<ncclComm_t[]> sample_nccl_comm;
  std::unique_ptr<ncclComm_t[]> load_nccl_comm;
  int sample_worker_num;
  int load_worker_num;
  std::unique_ptr<Coordinator> coordinator;
  CommInfo comm_info;
  CommInfo comm_info_load;
  std::unique_ptr<Coordinator> comm_coordinator;

  static DSContext* Global() {
    static DSContext instance;
    return &instance;
  }
};

}
}

#endif