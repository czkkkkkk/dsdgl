#ifndef DGL_DS_CONTEXT_H_
#define DGL_DS_CONTEXT_H_

#include <nccl.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <memory>

#include "coordinator.h"
#include "./comm/comm_info.h"
#include <atomic>
#include <vector>

using namespace dgl::runtime;

namespace dgl {
namespace ds {

struct DSContext {
  bool initialized = false;
  int world_size;
  int rank;
  int thread_num;
  std::vector<ncclComm_t> nccl_comm;
  std::vector<std::unique_ptr<CommInfo> > comm_info;
  std::unique_ptr<Coordinator> coordinator;
  std::unique_ptr<Coordinator> comm_coordinator;

  static DSContext* Global() {
    static DSContext instance;
    return &instance;
  }
};

}
}

#endif