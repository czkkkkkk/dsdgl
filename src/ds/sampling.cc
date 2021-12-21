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
// using namespace dgl::aten;

namespace dgl {
namespace ds {

void Sample(
    const HeteroGraphPtr hg,
    uint64 num_frontier,
    uint64 *frontier,
    int fanout,
    bool replace,
    uint64 **out_index) {
  assert(hg->NumEdgeTypes() == 1);
  dgl_type_t etype = 0;
  CSRMatrix mat = hg->GetCSRMatrix(etype);
  uint64 *in_ptr = static_cast<uint64*>(mat.indptr->data);
  uint64 *in_index = static_cast<uint64*>(mat.indices->data);
  SampleNeighbors(fanout, num_frontier, frontier, in_ptr, in_index, out_index);
}

DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  IdArray min_ids = args[1];
  IdArray nodes = args[2];
  int fanout = args[3];
  const std::string dir_str = args[4];
  const auto& prob = ListValueToVector<FloatArray>(args[5]);
  const bool replace = args[6];
  DSContextRef context_ref = args[7];
  auto* context = context_ref->GetContext();

  uint64 num_devices = min_ids->shape[0] - 1;
  uint64 *d_device_vids = static_cast<uint64*>(min_ids->data);
  uint64 num_seeds = nodes->shape[0];
  uint64 *d_seeds = static_cast<uint64*>(nodes->data); //local id
  uint64 h_device_col_ptr[num_devices + 1]; 
  uint64 h_device_col_cnt[num_devices];
  uint64 *d_device_col_cnt = nullptr;
  ConvertLidToGid(num_seeds, d_seeds, d_device_vids, context->rank);
  Cluster(num_devices, d_device_vids, num_seeds, d_seeds, fanout, 
          h_device_col_ptr, h_device_col_cnt, &d_device_col_cnt, &d_seeds_global);
  
  //convert local id to global id done in shuffle
  uint64 num_frontier;
  uint64 *d_frontier;
  uint64 h_device_offset[num_devices + 1];
  Shuffle(num_devices, h_device_col_ptr, h_device_col_cnt, d_device_col_cnt,
          d_seeds_global, num_frontier, h_device_offset, &d_frontier,
          context->rank, context->nccl_comm);

  //convert global id to local id done in sampling
  uint64 *d_local_out_cols;
  ConvertGidToLid(num_frontier, frontier, d_device_vids, context->rank);
  Sample(hg, num_frontier, frontier, fanout, replace, &d_local_out_cols);
  //todo convert local id to global id
  
  //final result
  uint64 *d_out_ptr, *d_global_out_cols;
  ConvertLidToGid(num_frontier * fanout, d_local_out_cols, d_device_vids, context->rank);
  Reshuffle(fanout, num_devices, h_device_offset, d_local_out_cols, h_device_col_ptr,
            num_seeds, d_out_ptr, d_global_out_cols, context->rank, context->nccl_comm);

  //现在的结果存成了csr，存储在d_out_ptr，d_global_out_cols里
  std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
  // *subg = ds::SampleNeighbors(
  //     hg.sptr(), min_ids, nodes, fanout, dir_str, prob, replace, context);
  *rv = HeteroSubgraphRef(subg);
});

}
}