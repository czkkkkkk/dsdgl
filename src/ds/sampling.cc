#include <nccl.h>
#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/sampling/neighbor.h>
#include "../c_api_common.h"
#include "../graph/unit_graph.h"
#include "context.h"
#include "cuda/ds_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace ds {

void Sample(
    const HeteroGraphPtr hg,
    uint64 num_frontier,
    uint64 *frontier,
    int fanout,
    bool replace,
    uint64 **out_index,
    uint64 **out_edges,
    int rank) {
  assert(hg->NumEdgeTypes() == 1);
  dgl_type_t etype = 0;
  CSRMatrix mat = hg->GetCSRMatrix(etype);
  uint64 *in_ptr = static_cast<uint64*>(mat.indptr->data);
  uint64 *in_index = static_cast<uint64*>(mat.indices->data);
  uint64 *edge_index = CSRHasData(mat) ? static_cast<uint64*>(mat.data->data) : nullptr;
  assert(edge_index != nullptr);
  // printf("rows: %d, cols: %d, data: %d\n", 
  //           int(mat.indptr->shape[0]), int(mat.indices->shape[0]), int(mat.data->shape[0]));
  // int64_t *rowp = new int64_t[mat.indptr->shape[0]];
  // int64_t *index = new int64_t[mat.indices->shape[0]];
  // int64_t *edges = new int64_t[mat.data->shape[0]];
  // CUDACHECK(cudaMemcpy(rowp, in_ptr, sizeof(int64_t)*(mat.indptr->shape[0]), cudaMemcpyDeviceToHost));
  // printf("copy1\n");
  // CUDACHECK(cudaMemcpy(index, in_index, sizeof(int64_t)*(mat.indices->shape[0]), cudaMemcpyDeviceToHost));
  // printf("copy2\n");
  // CUDACHECK(cudaMemcpy(edges, edge_index, sizeof(int64_t)*(mat.data->shape[0]), cudaMemcpyDeviceToHost));
  // delete[] rowp;
  // delete[] index;
  // delete[] edges;
  SampleNeighbors(fanout, num_frontier, frontier, in_ptr, in_index, edge_index, out_index, out_edges, rank);
}

DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  uint64_t num_vertices = args[1];
  IdArray min_vids = args[2];
  IdArray min_eids = args[3];
  IdArray nodes = args[4];
  int fanout = args[5];
  const std::string dir_str = args[6];
  const auto& prob = ListValueToVector<FloatArray>(args[7]);
  const bool replace = args[8];
  DSContextRef context_ref = args[9];
  auto* context = context_ref->GetContext();

  uint64 num_devices = min_vids->shape[0] - 1;
  uint64 *d_device_vids = static_cast<uint64*>(min_vids->data);
  uint64 *d_device_eids = static_cast<uint64*>(min_eids->data);
  uint64 num_seeds = nodes->shape[0];
  uint64 *d_seeds = static_cast<uint64*>(nodes->data); //local id
  uint64 *h_device_col_ptr = new uint64[num_devices + 1];
  uint64 *h_device_col_cnt = new uint64[num_devices];
  uint64 *d_device_col_cnt = nullptr;
  ConvertLidToGid(num_seeds, d_seeds, d_device_vids, context->rank);
  Cluster(num_devices, d_device_vids, num_seeds, d_seeds,
          h_device_col_ptr, h_device_col_cnt, &d_device_col_cnt);

  uint64 num_frontier;
  uint64 *d_frontier;
  uint64 *h_device_offset = new uint64[num_devices + 1];
  Shuffle(num_devices, h_device_col_ptr, h_device_col_cnt, d_device_col_cnt,
          d_seeds, num_frontier, h_device_offset, &d_frontier,
          context->rank, context->nccl_comm);

  //convert global id to local id done in sampling
  uint64 *d_local_out_cols, *d_local_out_edges;
  ConvertGidToLid(num_frontier, d_frontier, d_device_vids, context->rank);
  Sample(hg.sptr(), num_frontier, d_frontier, fanout, replace, &d_local_out_cols, &d_local_out_edges, context->rank);
  printf("finish sampling\n");

  // final result
  DLContext ctx;
  ctx.device_type = kDLGPU;
  ctx.device_id = context->rank; 
  IdArray out_ptr = NewIdArray(num_seeds + 1, ctx, sizeof(uint64) * 8);
  IdArray global_out_cols = NewIdArray(num_seeds * fanout, ctx, sizeof(uint64) * 8);
  IdArray global_out_edges = NewIdArray(num_seeds * fanout, ctx, sizeof(uint64) * 8);
  uint64 *d_out_ptr = static_cast<uint64*>(out_ptr->data); 
  uint64 *d_global_out_cols = static_cast<uint64*>(global_out_cols->data); 
  uint64 *d_global_out_edges = static_cast<uint64*>(global_out_edges->data);
  ConvertLidToGid(num_frontier * fanout, d_local_out_cols, d_device_vids, context->rank);
  ConvertLidToGid(num_frontier * fanout, d_local_out_edges, d_device_eids, context->rank);
  Reshuffle(fanout, num_devices, h_device_offset, d_local_out_cols, d_local_out_edges, h_device_col_ptr,
           num_seeds, d_out_ptr, d_global_out_cols, d_global_out_edges, context->rank, context->nccl_comm);
  
  printf("procedure complete\n");
  HeteroGraphPtr subg = UnitGraph::CreateFromCSR(1, num_seeds, num_seeds, 
                                                 out_ptr, global_out_cols, global_out_edges);
  *rv = HeteroGraphRef(subg);
});

// DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSSampleNeighbors")
// .set_body([] (DGLArgs args, DGLRetValue *rv) {
//   HeteroGraphRef hg = args[0];
//   uint64_t num_vertices = args[1];
//   IdArray min_vids = args[2];
//   IdArray min_eids = args[3];
//   IdArray nodes = args[4];
//   int fanout = args[5];
//   const std::string dir_str = args[6];
//   const auto& prob = ListValueToVector<FloatArray>(args[7]);
//   const bool replace = args[8];
//   DSContextRef context_ref = args[9];
//   auto* context = context_ref->GetContext();

//   uint64 num_devices = min_vids->shape[0] - 1;
//   uint64 *d_device_vids = static_cast<uint64*>(min_vids->data);
//   uint64 num_seeds = nodes->shape[0];
//   uint64 *d_seeds = static_cast<uint64*>(nodes->data); //local id
//   uint64 *h_device_col_ptr = new uint64[num_devices + 1];
//   uint64 *h_device_col_cnt = new uint64[num_devices];
//   uint64 *d_device_col_cnt = nullptr;
//   ConvertLidToGid(num_seeds, d_seeds, d_device_vids, context->rank);
//   Cluster(num_devices, d_device_vids, num_seeds, d_seeds,
//           h_device_col_ptr, h_device_col_cnt, &d_device_col_cnt);

//   uint64 num_frontier;
//   uint64 *d_frontier;
//   uint64 *h_device_offset = new uint64[num_devices + 1];
//   Shuffle(num_devices, h_device_col_ptr, h_device_col_cnt, d_device_col_cnt,
//           d_seeds, num_frontier, h_device_offset, &d_frontier,
//           context->rank, context->nccl_comm);

//   //convert global id to local id done in sampling
//   uint64 *d_local_out_cols, *d_local_out_edges;
//   ConvertGidToLid(num_frontier, d_frontier, d_device_vids, context->rank);
//   Sample(hg.sptr(), num_frontier, d_frontier, fanout, replace, &d_local_out_cols, &d_local_out_edges);
  
//   //final result
//   uint64 *d_out_ptr, *d_global_out_cols, *d_global_out_edges;
//   ConvertLidToGid(num_frontier * fanout, d_local_out_cols, d_device_vids, context->rank);
//   ConvertLidToGid(num_frontier * fanout, d_local_out_edges, d_device_vids, context->rank);
//   Reshuffle(fanout, num_devices, h_device_offset, d_local_out_cols, d_local_out_edges, h_device_col_ptr,
//             num_seeds, &d_out_ptr, &d_global_out_cols, &d_global_out_edges, context->rank, context->nccl_comm);
  
  
//   HeteroGraphPtr subg = UnitGraph::CreateFromCSR(1, num_vertices, num_vertices, d_out_ptr, d_global_out_cols, d_global_out_edges);
//   *rv = HeteroGraphRef(subg);
// });

}
}