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
#include "./cuda/alltoall.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace ds {

void Check(IdArray array, IdType limit) {
  int size = array->shape[0];
  IdType *data = array.Ptr<IdType>();
  IdType *hdata = new IdType[size];
  CUDACHECK(cudaMemcpy(hdata, data, sizeof(IdType) * size, cudaMemcpyDeviceToHost));
  for (int i=0; i<size; i++) {
    assert(hdata[i] < limit);
  }
  delete[] hdata;
}

HeteroGraphPtr CreateCOO(IdType num_vertices, IdArray seeds, int fanout, IdArray dst) {
  IdArray src;
  Replicate(seeds, &src, fanout);
  return UnitGraph::CreateFromCOO(1, num_vertices, num_vertices, dst, src);
}

void Show(IdArray array, int rank) {
  IdArray host_array = array.CopyTo(DLContext({kDLCPU, 0}));
  printf("rank %d ", rank);
  IdType *dst = host_array.Ptr<IdType>();
  for (int i=0; i<host_array->shape[0]; i++) {
    printf("%d ", dst[i]);
  }
  printf("\n");
}

DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  IdType num_vertices = args[1];
  IdArray min_vids = args[2];
  IdArray min_eids = args[3];
  IdArray seeds = args[4];
  int fanout = args[5];
  const std::string dir_str = args[6];
  const auto& prob = ListValueToVector<FloatArray>(args[7]);
  const bool replace = args[8];
  IdArray global_nid_map = args[9];
  const bool is_local = args[10];
  const bool bias = args[11];
  IdArray weight = args[12];
  auto* context = DSContext::Global();
  CHECK(context->graph_loaded);

  int n_seeds = seeds->shape[0];
  int rank = context->rank;
  int world_size = context->world_size;
  CUDACHECK(cudaSetDevice(rank));
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t s = thr_entry->stream;

  const DLContext& dgl_context = seeds->ctx;
  auto device = runtime::DeviceAPI::Get(dgl_context);
  if (dgl_context.device_type != DLDeviceType::kDLGPU) {
    LOG(FATAL) << "Seeds are not on GPUs";
  }

  if (is_local) {
    seeds = Partition(seeds, min_vids);
    CUDACHECK(cudaStreamSynchronize(s));
  }
  
  IdArray send_sizes, send_offset;
  Cluster(rank, seeds, min_vids, world_size, &send_sizes, &send_offset);
  if(context->enable_profiler) {
    CUDACHECK(cudaStreamSynchronize(s));
    context->profiler->UpdateDSSamplingNvlinkCount(send_offset, fanout);
  }

  IdArray frontier, recv_offset;
  std::tie(frontier, recv_offset) = Alltoall(seeds, send_offset, 1, rank, world_size);

  ConvertGidToLid(frontier, min_vids, rank);
  auto neighbors = SampleNeighbors(frontier, fanout, weight, bias);
  
  IdArray reshuffled_neighbors, reshuffle_recv_offset;
  std::tie(reshuffled_neighbors, reshuffle_recv_offset) = Alltoall(neighbors, recv_offset, fanout, rank, world_size, send_offset);

  HeteroGraphPtr subg = CreateCOO(num_vertices, seeds, fanout, reshuffled_neighbors);
  
  List<ObjectRef> ret;
  ret.push_back(HeteroGraphRef(subg));
  ret.push_back(Value(MakeValue(seeds)));
  *rv = ret;
  // *rv = HeteroGraphRef(subg);
  CUDACHECK(cudaStreamSynchronize(s));
});

IdArray ToGlobal(IdArray nids, IdArray global_nid_map) {
  CHECK(nids->ctx.device_type == kDLCPU);
  CHECK(global_nid_map->ctx.device_type == kDLCPU);
  IdType length = nids->shape[0];
  IdArray ret = IdArray::Empty({length}, nids->dtype, nids->ctx);
  IdType* ret_ptr = ret.Ptr<IdType>();
  IdType* nids_ptr = nids.Ptr<IdType>();
  IdType* global_nid_map_ptr = global_nid_map.Ptr<IdType>();
  for(int i = 0; i < length; ++i) {
    ret_ptr[i] = global_nid_map_ptr[nids_ptr[i]];
  }
  return ret;
}

void ToGlobalInplace(IdArray nids, IdArray global_nid_map) {
  CHECK(nids->ctx.device_type == kDLCPU);
  CHECK(global_nid_map->ctx.device_type == kDLCPU);
  IdType length = nids->shape[0];
  IdArray ret = IdArray::Empty({length}, nids->dtype, nids->ctx);
  IdType* ret_ptr = ret.Ptr<IdType>();
  IdType* nids_ptr = nids.Ptr<IdType>();
  IdType* global_nid_map_ptr = global_nid_map.Ptr<IdType>();
  for(int i = 0; i < length; ++i) {
    nids_ptr[i] = global_nid_map_ptr[nids_ptr[i]];
  }
}

IdArray Rebalance(IdArray ids, int batch_size, Coordinator* coor) {
  auto ids_vec = ids.ToVector<int64_t>();
  auto vecs = coor->Gather(ids_vec);
  if(coor->IsRoot()) {
    int total = 0;
    for (const auto& vec: vecs) {
      total += vec.size();
    }
    int world_size = coor->GetWorldSize();
    int batch_per_rank = total / (world_size * batch_size);
    int size_per_rank = batch_per_rank * batch_size;
    for(auto& vec: vecs) {
      if(vec.size() > size_per_rank) {
        auto redundant = std::vector<int64_t>(vec.begin() + size_per_rank, vec.end());
        vec.resize(size_per_rank);
        int cur_rank = 0;
        for(int i = 0; i < redundant.size(); ++i) {
          int loop_size = 0;
          while(loop_size < world_size) {
            if(vecs[cur_rank].size() < size_per_rank) {
              vecs[cur_rank].push_back(redundant[i]);
              break;
            }
            cur_rank = (cur_rank + 1) % world_size;
            loop_size++;
          }
          if(loop_size == world_size) {
            break;
          }
        }
      }
    }
    for(auto& vec: vecs) {
      std::random_shuffle(vec.begin(), vec.end());
    }
  }
  auto ret = coor->Scatter(vecs);
  return NDArray::FromVector(ret);
}

IdArray RebalanceSimple(IdArray ids, int batch_size, Coordinator* coor) {
  auto ids_vec = ids.ToVector<int64_t>();
  auto vecs = coor->Gather(ids_vec);
  if(coor->IsRoot()) {
    int total = 0;
    int min_size = vecs[0].size();
    for (const auto& vec: vecs) {
      total += vec.size();
      min_size = std::min(min_size, (int)vecs[0].size());
    }
    min_size = min_size - min_size % batch_size;
    for(auto& vec: vecs) {
      vec.resize(min_size);
    }
    for(auto& vec: vecs) {
      std::random_shuffle(vec.begin(), vec.end());
    }
  }
  auto ret = coor->Scatter(vecs);
  return NDArray::FromVector(ret);
}

IdArray RebalanceRandom(IdArray ids, int batch_size, Coordinator* coor) {
  auto ids_vec = ids.ToVector<int64_t>();
  auto vecs = coor->Gather(ids_vec);
  if(coor->IsRoot()) {
    int total = 0;
    for (const auto& vec: vecs) {
      total += vec.size();
    }
    int world_size = coor->GetWorldSize();
    auto flatten = Flatten(vecs);
    std::random_shuffle(flatten.begin(), flatten.end());
    int size_per_rank = total / world_size;
    flatten.resize(size_per_rank * world_size);
    for(int i = 0; i < world_size; ++i) {
      vecs[i] = std::vector<int64_t>(flatten.begin() + i * size_per_rank, flatten.begin() + (i + 1) * size_per_rank);
    }
  }
  auto ret = coor->Scatter(vecs);
  return NDArray::FromVector(ret);
}

/**
 * @brief Rebalance local node ids of all ranks so that each rank have
 * the same number of node ids. It may drop some node ids to keep balance.
 * Note that the output are global node ids.
 * 
 * @param nids local node ids
 * @param batch_size pack as much ids as possible according to the batch_size
 * @param global_nid_map
 * 
 * @return balanced global node ids
 */
DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSRebalanceNIds")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  IdArray nids = args[0];
  int batch_size = args[1];
  IdArray global_nid_map = args[2];
  IdArray global_nids = ToGlobal(nids, global_nid_map);
  LOG(INFO) << "Rank: " << DSContext::Global()->rank << ", # train ids before rebalance: " << nids->shape[0];
  auto* coor = DSContext::Global()->coordinator.get();
  auto ret = RebalanceRandom(global_nids, batch_size, coor);
  *rv = ret;
});

DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSCSRToGlobalId")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  IdArray global_nid_map = args[1];
  assert(hg->NumEdgeTypes() == 1);
  dgl_type_t etype = 0;
  CSRMatrix csr_mat = hg->GetCSRMatrix(etype);
  // csr_mat.indices = ToGlobal(csr_mat.indices, global_nid_map);
  ToGlobalInplace(csr_mat.indices, global_nid_map);
  *rv = hg;
});

void SampleUVA(IdArray frontier, IdArray row_idx, const HeteroGraphPtr hg, int fanout, bool replace, IdArray* neighbors, IdArray* edges) {
  assert(hg->NumEdgeTypes() == 1);
  dgl_type_t etype = 0;
  CSRMatrix csr_mat = hg->GetCSRMatrix(etype);
  SampleNeighborsUVA(frontier, row_idx, csr_mat, fanout, neighbors, edges);
}

DGL_REGISTER_GLOBAL("ds.sampling._CAPI_DGLDSSampleNeighborsUVA")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  // IdArray row_idx = args[0];
  HeteroGraphRef hg = args[0];
  IdArray seeds = args[1];
  int fanout = args[2];
  bool replace = args[3];
  IdArray neighbors, edges;
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  cudaStream_t s = thr_entry->stream;
  auto row_idx = hg.sptr()->GetCSRMatrix(0).indptr;
  SampleUVA(seeds, row_idx, hg.sptr(), fanout, replace, &neighbors, &edges);
  CUDACHECK(cudaStreamSynchronize(s));
  IdType num_vertices = row_idx->shape[0];
  HeteroGraphPtr subg = CreateCOO(num_vertices, seeds, fanout, neighbors);
  *rv = HeteroGraphRef(subg);
});

}
}
