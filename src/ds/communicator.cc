#include "nccl.h"
#include "../c_api_common.h"
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include "dmlc/logging.h"
#include <cuda_runtime.h>

#include "context.h"


using namespace dgl::runtime;

namespace dgl {
namespace ds {

DGL_REGISTER_GLOBAL("ds._CAPI_DGLNCCLGetUniqueId")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    int rank = args[0];
    ncclUniqueId id;
    if(rank == 0) {
      ncclGetUniqueId(&id);
    } 
    auto id_array = DGLByteArray();
    id_array.data = id.internal;
    id_array.size = NCCL_UNIQUE_ID_BYTES;
    *rv = id_array;
  });
DGL_REGISTER_GLOBAL("ds._CAPI_DGLNCCLInit")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    std::string id_str = args[0];
    int rank = args[1];
    int world_size = args[2];
    auto* context = DSContext::Global();

    context->initialized = true;
    context->world_size = world_size;
    context->rank = rank;

    ncclUniqueId nccl_id;
    memcpy(nccl_id.internal, id_str.c_str(), NCCL_UNIQUE_ID_BYTES);
    cudaSetDevice(rank);
    if(world_size > 1) {
      ncclCommInitRank(&context->nccl_comm, world_size, nccl_id, rank);
    }
    LOG(INFO) << "Rank " + std::to_string(rank) + " successfully build nccl communicator";
  });

}
}