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
#include "./utils.h"
#include <assert.h>
// #include "cuda/test.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace ds {


DGL_REGISTER_GLOBAL("ds.pin_graph._CAPI_DGLDSPinGraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  int rank = args[1];
  HeteroGraphPtr hg_ptr = hg.sptr();
  CSRMatrix mat = hg_ptr->GetCSRMatrix(0);
  mat.indptr = CreateShmArray(mat.indptr, "uva_graph_indptr");
  mat.indices = CreateShmArray(mat.indices, "uva_graph_indices");
  int64_t n_vertices = hg_ptr->GetRelationGraph(0)->NumVertices(0);
  int64_t n_edges = hg_ptr->GetRelationGraph(0)->NumEdges(0);
  auto edge_ids = Range(0, n_edges, 64, {kDLGPU, rank});
  auto ug = CreateFromCSR(1, n_vertices, n_vertices, mat.indptr, mat.indices, edge_ids);
  auto new_hg = HeteroGraphPtr(new HeteroGraph(hg_ptr->meta_graph(), {ug}, {n_vertices}));
  *rv = HeteroGraphRef(new_hg);
});

}
}