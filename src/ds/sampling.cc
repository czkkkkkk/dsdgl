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

void Sample(IdArray frontier, const HeteroGraphPtr hg, int fanout, bool replace, IdArray* neighbors, IdArray* edges) {
  assert(hg->NumEdgeTypes() == 1);
  dgl_type_t etype = 0;
  CSRMatrix csr_mat = hg->GetCSRMatrix(etype);
  SampleNeighbors(frontier, csr_mat, fanout, neighbors, edges);
}

void Check(IdArray array, uint64 limit) {
  int size = array->shape[0];
  uint64 *data = array.Ptr<IdType>();
  uint64 *hdata = new uint64[size];
  CUDACHECK(cudaMemcpy(hdata, data, sizeof(uint64) * size, cudaMemcpyDeviceToHost));
  for (int i=0; i<size; i++) {
    assert(hdata[i] < limit);
  }
  delete[] hdata;
}

HeteroGraphPtr CreateCOO(uint64_t num_vertices, IdArray seeds, int fanout, IdArray dst) {
  IdArray src;
  Replicate(seeds, &src, fanout);
  return UnitGraph::CreateFromCOO(1, num_vertices, num_vertices, src, dst);
}

void Show(IdArray array, int rank) {
  IdArray host_array = array.CopyTo(DLContext({kDLCPU, 0}));
  printf("rank %d ", rank);
  uint64 *dst = host_array.Ptr<uint64>();
  for (int i=0; i<host_array->shape[0]; i++) {
    printf("%d ", dst[i]);
  }
  printf("\n");
}

DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  uint64_t num_vertices = args[1];
  IdArray min_vids = args[2];
  IdArray min_eids = args[3];
  IdArray seeds = args[4];
  int fanout = args[5];
  const std::string dir_str = args[6];
  const auto& prob = ListValueToVector<FloatArray>(args[7]);
  const bool replace = args[8];
  IdArray global_nid_map = args[9];
  const bool is_local = args[10];
  auto* context = DSContext::Global();

  int n_seeds = seeds->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;

  const DLContext& dgl_context = seeds->ctx;
  auto device = runtime::DeviceAPI::Get(dgl_context);
  if (dgl_context.device_type != DLDeviceType::kDLGPU) {
    LOG(FATAL) << "Seeds are not on GPUs";
  }

  if (is_local) {
    ConvertLidToGid(seeds, global_nid_map);
  }

  IdArray send_sizes, send_offset;
  Cluster(seeds, min_vids, world_size, &send_sizes, &send_offset);
  auto host_send_sizes = send_sizes.CopyTo(DLContext({kDLCPU, 0}));
  auto host_send_offset = send_offset.CopyTo(DLContext({kDLCPU, 0}));

  IdArray frontier, host_recv_offset;
  Shuffle(seeds, host_send_offset, send_sizes, rank, world_size, context->nccl_comm, &frontier, &host_recv_offset);

  ConvertGidToLid(frontier, min_vids, rank);
  IdArray neighbors, edges;
  Sample(frontier, hg.sptr(), fanout, replace, &neighbors, &edges);

  ConvertLidToGid(neighbors, global_nid_map);
  
  IdArray reshuffled_neighbors;
  Reshuffle(neighbors, fanout, n_seeds, host_send_offset, host_recv_offset, rank, world_size, context->nccl_comm, &reshuffled_neighbors);
  // LOG(INFO) << "Reshuffled neibhgors: " << ToDebugString(reshuffled_neighbors);
  
  // ConvertGidToLid(seeds, min_vids, rank);
  HeteroGraphPtr subg = CreateCOO(num_vertices, seeds, fanout, reshuffled_neighbors);

  *rv = HeteroGraphRef(subg);
});

}
}