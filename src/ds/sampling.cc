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
    uint64 **out_edges) {
  assert(hg->NumEdgeTypes() == 1);
  dgl_type_t etype = 0;
  CSRMatrix mat = hg->GetCSRMatrix(etype);
  uint64 *in_ptr = static_cast<uint64*>(mat.indptr->data);
  uint64 *in_index = static_cast<uint64*>(mat.indices->data);
  uint64 *edge_index = CSRHasData(mat) ? static_cast<uint64*>(mat.data->data) : nullptr;
  assert(edge_index != nullptr);
  // printf("rows: %d, cols: %d\n", int(mat.num_rows), int(mat.num_cols));
  SampleNeighbors(fanout, num_frontier, frontier, in_ptr, in_index, edge_index, out_index, out_edges);
}

DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  IdArray min_vids = args[1];
  IdArray min_eids = args[2];
  IdArray nodes = args[3];
  int fanout = args[4];
  const std::string dir_str = args[5];
  const auto& prob = ListValueToVector<FloatArray>(args[6]);
  const bool replace = args[7];
  DSContextRef context_ref = args[8];
  auto* context = context_ref->GetContext();

  uint64 num_devices = min_vids->shape[0] - 1;
  uint64 *d_device_vids = static_cast<uint64*>(min_vids->data);
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
  Sample(hg.sptr(), num_frontier, d_frontier, fanout, replace, &d_local_out_cols, &d_local_out_edges);
  
  //final result
  uint64 *d_out_ptr, *d_global_out_cols, *d_global_out_edges;
  ConvertLidToGid(num_frontier * fanout, d_local_out_cols, d_device_vids, context->rank);
  ConvertLidToGid(num_frontier * fanout, d_local_out_edges, d_device_vids, context->rank);
  Reshuffle(fanout, num_devices, h_device_offset, d_local_out_cols, d_local_out_edges, h_device_col_ptr,
            num_seeds, &d_out_ptr, &d_global_out_cols, &d_global_out_edges, context->rank, context->nccl_comm);

  //现在的结果存成了csr，存储在d_out_ptr，d_global_out_cols里
  std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
  // *subg = ds::SampleNeighbors(
  //     hg.sptr(), min_ids, nodes, fanout, dir_str, prob, replace, context);
  *rv = HeteroSubgraphRef(subg);
});

}
}