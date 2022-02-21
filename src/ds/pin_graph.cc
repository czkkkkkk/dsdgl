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
#include "./memory_manager.h"
#include <assert.h>
// #include "cuda/test.h"

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

void Register(IdArray array) {
  uint64_t *data = array.Ptr<uint64_t>();
  uint64_t size = array->shape[0];
  assert(data != nullptr);
  assert(size > 0);
  printf("graph size: %lu\n", size);
  CUDACHECK(cudaHostRegister((void*)data, sizeof(uint64_t) * size, cudaHostRegisterMapped));
  CUDACHECK(cudaDeviceSynchronize());
}

DGL_REGISTER_GLOBAL("ds.pin_graph._CAPI_DGLDSPinGraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  int rank = args[1];
  HeteroGraphPtr hg_ptr = hg.sptr();
  CSRMatrix mat = hg_ptr->GetCSRMatrix(0);
  IdArray indptr = mat.indptr;
  IdArray ret = IdArray::Empty({indptr->shape[0]}, indptr->dtype, DLContext({kDLGPU, rank}));
  CUDACHECK(cudaMemcpy(ret.Ptr<uint64_t>(), indptr.Ptr<uint64_t>(), sizeof(uint64_t) * indptr->shape[0], cudaMemcpyHostToDevice));
  CUDACHECK(cudaDeviceSynchronize());
  IdArray indices = mat.indices;
  printf("vertex number: %lu, edge number: %lu\n", indptr->shape[0], indices->shape[0]);
  Register(indices);
  *rv = ret;
});

/*
DGL_REGISTER_GLOBAL("ds.pin_graph._CAPI_DGLDSPinGraphTest")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  IdArray array = args[0];
  const DLContext& dgl_context = array->ctx;
  auto device = runtime::DeviceAPI::Get(dgl_context);
  assert(dgl_context.device_type == kDLGPU);
  printf("rank: %d\n", device);

  uint64_t *data = array.Ptr<uint64_t>();
  uint64_t size = array->shape[0];
  TestInc(data, size);
  CUDACHECK(cudaDeviceSynchronize());
});
*/

}
}