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
#include "../graph/heterograph.h"
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
  mat.indptr = CreateShmArray(mat.indptr, mat.indptr->shape[0], "uva_graph_indptr");
  mat.indices = CreateShmArray(mat.indices, mat.indices->shape[0], "uva_graph_indices");
  int64_t n_vertices = hg_ptr->GetRelationGraph(0)->NumVertices(0);
  int64_t n_edges = hg_ptr->GetRelationGraph(0)->NumEdges(0);
  auto edge_ids = Range(0, n_edges, 64, {kDLGPU, rank});
  auto ug = CreateFromCSR(1, n_vertices, n_vertices, mat.indptr, mat.indices, edge_ids);
  auto new_hg = HeteroGraphPtr(new HeteroGraph(hg_ptr->meta_graph(), {ug}, {n_vertices}));
  *rv = HeteroGraphRef(new_hg);
});

}
}