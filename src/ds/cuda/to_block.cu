#include <dgl/runtime/device_api.h>
#include <dgl/immutable_graph.h>
#include <cuda_runtime.h>
#include <utility>
#include <algorithm>
#include <memory>

#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_hashtable.cuh"
#include "../../graph/heterograph.h"

#include "../memory_manager.h"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;

namespace dgl {

namespace ds {

template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__device__ void map_vertex_ids(
    const IdType * const global,
    IdType * const new_global,
    const IdType num_vertices,
    const DeviceOrderedHashTable<IdType>& table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  const IdType tile_start = TILE_SIZE*blockIdx.x;
  const IdType tile_end = min(TILE_SIZE*(blockIdx.x+1), num_vertices);

  for (IdType idx = threadIdx.x+tile_start; idx < tile_end; idx+=BLOCK_SIZE) {
    const Mapping& mapping = *table.Search(global[idx]);
    new_global[idx] = mapping.local;
  }
}

template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_edge_ids(
    const IdType * const global_srcs_device,
    IdType * const new_global_srcs_device,
    const IdType * const global_dsts_device,
    IdType * const new_global_dsts_device,
    const IdType num_edges,
    DeviceOrderedHashTable<IdType> src_mapping,
    DeviceOrderedHashTable<IdType> dst_mapping) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_srcs_device,
        new_global_srcs_device,
        num_edges,
        src_mapping);
  } else {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_dsts_device,
        new_global_dsts_device,
        num_edges,
        dst_mapping);
  }
}

template<typename IdType>
inline size_t RoundUpDiv(
    const IdType num,
    const size_t divisor) {
  return static_cast<IdType>(num/divisor) + (num % divisor == 0 ? 0 : 1);
}


template<typename IdType>
inline IdType RoundUp(
    const IdType num,
    const size_t unit) {
  return RoundUpDiv(num, unit)*unit;
}


template<typename IdType>
class DeviceNodeMap {
 public:
  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  DeviceNodeMap(
      const std::vector<int64_t>& num_nodes,
      DGLContext ctx,
      cudaStream_t stream) :
    num_types_(num_nodes.size()),
    rhs_offset_(num_types_/2),
    workspaces_(),
    hash_tables_(),
    ctx_(ctx) {
    auto device = runtime::DeviceAPI::Get(ctx);

    hash_tables_.reserve(num_types_);
    workspaces_.reserve(num_types_);
    for (int64_t i = 0; i < num_types_; ++i) {
      hash_tables_.emplace_back(
          new OrderedHashTable<IdType>(
            num_nodes[i],
            ctx_,
            stream));
    }
  }

  OrderedHashTable<IdType>& LhsHashTable(
      const size_t index) {
    return HashData(index);
  }

  OrderedHashTable<IdType>& RhsHashTable(
      const size_t index) {
    return HashData(index+rhs_offset_);
  }

  const OrderedHashTable<IdType>& LhsHashTable(
      const size_t index) const {
    return HashData(index);
  }

  const OrderedHashTable<IdType>& RhsHashTable(
      const size_t index) const {
    return HashData(index+rhs_offset_);
  }

  IdType LhsHashSize(
      const size_t index) const {
    return HashSize(index);
  }

  IdType RhsHashSize(
      const size_t index) const {
    return HashSize(rhs_offset_+index);
  }

  size_t Size() const {
    return hash_tables_.size();
  }

 private:
  int64_t num_types_;
  size_t rhs_offset_;
  std::vector<void*> workspaces_;
  std::vector<std::unique_ptr<OrderedHashTable<IdType>>> hash_tables_;
  DGLContext ctx_;

  inline OrderedHashTable<IdType>& HashData(
      const size_t index) {
    CHECK_LT(index, hash_tables_.size());
    return *hash_tables_[index];
  }

  inline const OrderedHashTable<IdType>& HashData(
      const size_t index) const {
    CHECK_LT(index, hash_tables_.size());
    return *hash_tables_[index];
  }

  inline IdType HashSize(
      const size_t index) const {
    return HashData(index).size();
  }
};

template<typename IdType>
class DeviceNodeMapMaker {
 public:
  DeviceNodeMapMaker(
      const std::vector<int64_t>& maxNodesPerType) :
      max_num_nodes_(0) {
    max_num_nodes_ = *std::max_element(maxNodesPerType.begin(),
        maxNodesPerType.end());
  }

  /**
  * \brief This function builds node maps for each node type, preserving the
  * order of the input nodes.
  *
  * \param lhs_nodes The set of source input nodes.
  * \param rhs_nodes The set of destination input nodes.
  * \param node_maps The node maps to be constructed.
  * \param count_lhs_device The number of unique source nodes (on the GPU).
  * \param lhs_device The unique source nodes (on the GPU).
  * \param stream The stream to operate on.
  */
  void Make(
      const std::vector<IdArray>& lhs_nodes,
      const std::vector<IdArray>& rhs_nodes,
      DeviceNodeMap<IdType> * const node_maps,
      int64_t * const count_lhs_device,
      std::vector<IdArray>* const lhs_device,
      cudaStream_t stream) {
    const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

    CUDA_CALL(cudaMemsetAsync(
      count_lhs_device,
      0,
      num_ntypes*sizeof(*count_lhs_device),
      stream));

    // possibly dublicate lhs nodes
    const int64_t lhs_num_ntypes = static_cast<int64_t>(lhs_nodes.size());
    for (int64_t ntype = 0; ntype < lhs_num_ntypes; ++ntype) {
      const IdArray& nodes = lhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDLGPU);
        node_maps->LhsHashTable(ntype).FillWithDuplicates(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            (*lhs_device)[ntype].Ptr<IdType>(),
            count_lhs_device+ntype,
            stream);
      }
    }

    // unique rhs nodes
    const int64_t rhs_num_ntypes = static_cast<int64_t>(rhs_nodes.size());
    for (int64_t ntype = 0; ntype < rhs_num_ntypes; ++ntype) {
      const IdArray& nodes = rhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        node_maps->RhsHashTable(ntype).FillWithUnique(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            stream);
      }
    }
  }

 private:
  IdType max_num_nodes_;
};

template<typename IdType>
std::tuple<std::vector<IdArray>, std::vector<IdArray>>
MapEdges(
    HeteroGraphPtr graph,
    const std::vector<EdgeArray>& edge_sets,
    const DeviceNodeMap<IdType>& node_map,
    cudaStream_t stream) {
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;

  const auto& ctx = graph->Context();

  std::vector<IdArray> new_lhs;
  new_lhs.reserve(edge_sets.size());
  std::vector<IdArray> new_rhs;
  new_rhs.reserve(edge_sets.size());

  // The next peformance optimization here, is to perform mapping of all edge
  // types in a single kernel launch.
  const int64_t num_edge_sets = static_cast<int64_t>(edge_sets.size());
  for (int64_t etype = 0; etype < num_edge_sets; ++etype) {
    const EdgeArray& edges = edge_sets[etype];
    if (edges.id.defined() && edges.src->shape[0] > 0) {
      const int64_t num_edges = edges.src->shape[0];
      // new_lhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType)*8));
      // new_rhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType)*8));
      
      DLDataType dtype = edges.src->dtype;
      new_lhs.emplace_back(MemoryManager::Global()->Empty("NEW_LHS", {num_edges}, dtype, ctx));
      new_rhs.emplace_back(MemoryManager::Global()->Empty("NEW_RHS", {num_edges}, dtype, ctx));
      assert(new_lhs[etype]->shape[0] == num_edges);
      assert(new_rhs[etype]->shape[0] == num_edges);

      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const int src_type = src_dst_types.first;
      const int dst_type = src_dst_types.second;

      const dim3 grid(RoundUpDiv(num_edges, TILE_SIZE), 2);
      const dim3 block(BLOCK_SIZE);

      // map the srcs
      map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE><<<
        grid,
        block,
        0,
        stream>>>(
        edges.src.Ptr<IdType>(),
        new_lhs.back().Ptr<IdType>(),
        edges.dst.Ptr<IdType>(),
        new_rhs.back().Ptr<IdType>(),
        num_edges,
        node_map.LhsHashTable(src_type).DeviceHandle(),
        node_map.RhsHashTable(dst_type).DeviceHandle());
      CUDA_CALL(cudaGetLastError());
    } else {
      new_lhs.emplace_back(
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx));
      new_rhs.emplace_back(
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx));
    }
  }

  return std::tuple<std::vector<IdArray>, std::vector<IdArray>>(
      std::move(new_lhs), std::move(new_rhs));
}

template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBlockGPU(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    const bool include_rhs_in_lhs) {
  cudaStream_t stream = 0;
  const auto& ctx = graph->Context();
  auto device = runtime::DeviceAPI::Get(ctx);
  DLDataType dtype = rhs_nodes[0]->dtype;

  CHECK_EQ(ctx.device_type, kDLGPU);
  for (const auto& nodes : rhs_nodes) {
    CHECK_EQ(ctx.device_type, nodes->ctx.device_type);
  }

  // Since DST nodes are included in SRC nodes, a common requirement is to fetch
  // the DST node features from the SRC nodes features. To avoid expensive sparse lookup,
  // the function assures that the DST nodes in both SRC and DST sets have the same ids.
  // As a result, given the node feature tensor ``X`` of type ``utype``,
  // the following code finds the corresponding DST node features of type ``vtype``:

  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();

  assert(num_etypes == 1);
  assert(num_ntypes == 1);

  CHECK(rhs_nodes.size() == static_cast<size_t>(num_ntypes))
    << "rhs_nodes not given for every node type";

  std::vector<EdgeArray> edge_arrays(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t dsttype = src_dst_types.second;
    if (!aten::IsNullArray(rhs_nodes[dsttype])) {
      edge_arrays[etype] = graph->Edges(etype);
    }
  }

  // count lhs and rhs nodes
  // 前一半是lhs，后一半是rhs
  std::vector<int64_t> maxNodesPerType(num_ntypes*2, 0);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    maxNodesPerType[ntype+num_ntypes] += rhs_nodes[ntype]->shape[0];

    if (include_rhs_in_lhs) {
      maxNodesPerType[ntype] += rhs_nodes[ntype]->shape[0];
    }
  }
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    if (edge_arrays[etype].src.defined()) {
      maxNodesPerType[srctype] += edge_arrays[etype].src->shape[0];
    }
  }

  // gather lhs_nodes
  std::vector<int64_t> src_node_offsets(num_ntypes, 0);
  std::vector<IdArray> src_nodes(num_ntypes);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    // src_nodes[ntype] = NewIdArray(maxNodesPerType[ntype], ctx,
    //     sizeof(IdType)*8);
    
    src_nodes[ntype] = MemoryManager::Global()->Empty("SRC_NODES", 
              {maxNodesPerType[ntype]}, dtype, ctx);
    assert(src_nodes[ntype]->shape[0] == maxNodesPerType[ntype]);

    if (include_rhs_in_lhs) {
      // place rhs nodes first
      device->CopyDataFromTo(rhs_nodes[ntype].Ptr<IdType>(), 0,
          src_nodes[ntype].Ptr<IdType>(), src_node_offsets[ntype],
          sizeof(IdType)*rhs_nodes[ntype]->shape[0],
          rhs_nodes[ntype]->ctx, src_nodes[ntype]->ctx,
          rhs_nodes[ntype]->dtype,
          stream);
      src_node_offsets[ntype] += sizeof(IdType)*rhs_nodes[ntype]->shape[0];
    }
  }
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    if (edge_arrays[etype].src.defined()) {
      device->CopyDataFromTo(
          edge_arrays[etype].src.Ptr<IdType>(), 0,
          src_nodes[srctype].Ptr<IdType>(),
          src_node_offsets[srctype],
          sizeof(IdType)*edge_arrays[etype].src->shape[0],
          rhs_nodes[srctype]->ctx,
          src_nodes[srctype]->ctx,
          rhs_nodes[srctype]->dtype,
          stream);

      src_node_offsets[srctype] += sizeof(IdType)*edge_arrays[etype].src->shape[0];
    }
  }

  // allocate space for map creation process
  DeviceNodeMapMaker<IdType> maker(maxNodesPerType);

  DeviceNodeMap<IdType> node_maps(maxNodesPerType, ctx, stream);

  int64_t total_lhs = 0;
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    total_lhs += maxNodesPerType[ntype];
  }

  std::vector<IdArray> lhs_nodes;
  lhs_nodes.reserve(num_ntypes);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    // lhs_nodes.emplace_back(NewIdArray(
    //  maxNodesPerType[ntype], ctx, sizeof(IdType)*8));
    lhs_nodes.emplace_back(MemoryManager::Global()->Empty("LHS_NODES", 
       {maxNodesPerType[ntype]}, dtype, ctx));
    lhs_nodes[ntype]->shape[0] = maxNodesPerType[ntype];
    assert(lhs_nodes[ntype]->shape[0] == maxNodesPerType[ntype]);
  }

  // populate the mappings
  int64_t * count_lhs_device = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*num_ntypes*2));
  maker.Make(
      src_nodes,
      rhs_nodes,
      &node_maps,
      count_lhs_device,
      &lhs_nodes,
      stream);

  std::vector<IdArray> induced_edges;
  induced_edges.reserve(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].id.defined()) {
      induced_edges.push_back(edge_arrays[etype].id);
    } else {
      induced_edges.push_back(
            aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx));
    }
  }

  // build metagraph -- small enough to be done on CPU
  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, new_dst);

  // allocate vector for graph relations while GPU is busy
  std::vector<HeteroGraphPtr> rel_graphs;
  rel_graphs.reserve(num_etypes);

  std::vector<int64_t> num_nodes_per_type(num_ntypes*2);
  // populate RHS nodes from what we already know
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    num_nodes_per_type[num_ntypes+ntype] = rhs_nodes[ntype]->shape[0];
  }
  device->CopyDataFromTo(
      count_lhs_device, 0,
      num_nodes_per_type.data(), 0,
      sizeof(*num_nodes_per_type.data())*num_ntypes,
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, 64, 1},
      stream);
  device->StreamSync(ctx, stream);

  // wait for the node counts to finish transferring
  device->FreeWorkspace(ctx, count_lhs_device);

  // map node numberings from global to local, and build pointer for CSR
  std::vector<IdArray> new_lhs;
  std::vector<IdArray> new_rhs;
  std::tie(new_lhs, new_rhs) = MapEdges(graph, edge_arrays, node_maps, stream);

  // resize lhs nodes
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    lhs_nodes[ntype]->shape[0] = num_nodes_per_type[ntype];
  }

  // build the heterograph
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;

    if (rhs_nodes[dsttype]->shape[0] == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_nodes[srctype]->shape[0], rhs_nodes[dsttype]->shape[0],
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx),
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx)));
    } else {
      rel_graphs.push_back(CreateFromCOO(
          2,
          lhs_nodes[srctype]->shape[0],
          rhs_nodes[dsttype]->shape[0],
          new_lhs[etype],
          new_rhs[etype]));
    }
  }

  HeteroGraphPtr new_graph = CreateHeteroGraph(
      new_meta_graph, rel_graphs, num_nodes_per_type);

  // return the new graph, the new src nodes, and new edges
  return std::make_tuple(new_graph, lhs_nodes, induced_edges);
}

std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray> >
ToBlock(HeteroGraphPtr graph,
        const std::vector<IdArray> &rhs_nodes,
        bool include_rhs_in_lhs) {
  DLDataType dtype = rhs_nodes[0]->dtype;
  const auto& ctx = graph->Context();
  
  MemoryManager::Global()->Empty("NEW_LHS", {(1024*25+1024)*10}, dtype, ctx);
  MemoryManager::Global()->Empty("NEW_RHS", {(1024*25+1024)*10}, dtype, ctx);
  MemoryManager::Global()->Empty("SRC_NODES", {(1024*25+1024)*10}, dtype, ctx);
  MemoryManager::Global()->Empty("LHS_NODES", {(1024*25+1024)*10}, dtype, ctx);
  MemoryManager::Global()->ClearUseCount();

  return ToBlockGPU<int64_t>(graph, rhs_nodes, include_rhs_in_lhs);
}

}
}