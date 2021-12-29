#ifndef DGL_DS_CONTEXT_H_
#define DGL_DS_CONTEXT_H_

#include <nccl.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <memory>

using namespace dgl::runtime;

namespace dgl {
namespace ds {

struct DSContext {
  bool initialized = false;
  int world_size;
  int rank;
  ncclComm_t nccl_comm;

  static DSContext* Global() {
    static DSContext instance;
    return &instance;
  }
};

}
}

#endif