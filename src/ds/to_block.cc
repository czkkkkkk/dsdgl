#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/container.h>
#include <vector>
#include <tuple>

#include "cuda/to_block.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace ds {

using IdType = uint64_t;

DGL_REGISTER_GLOBAL("ds.to_block._CAPI_DGLDSToBlock")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];
    const std::vector<IdArray> &rhs_nodes = ListValueToVector<IdArray>(args[1]);
    const bool include_rhs_in_lhs = args[2];

    HeteroGraphPtr new_graph;
    std::vector<IdArray> lhs_nodes;
    std::vector<IdArray> induced_edges;

    std::tie(new_graph, lhs_nodes, induced_edges) = ToBlock(graph_ref.sptr(), rhs_nodes, include_rhs_in_lhs);

    List<Value> lhs_nodes_ref;
    for (IdArray &array : lhs_nodes)
      lhs_nodes_ref.push_back(Value(MakeValue(array)));
    List<Value> induced_edges_ref;
    for (IdArray &array : induced_edges)
      induced_edges_ref.push_back(Value(MakeValue(array)));

    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(new_graph));
    ret.push_back(lhs_nodes_ref);
    ret.push_back(induced_edges_ref);

    *rv = ret;
  });

}

}

/*
void Make(IdArray lhs_nodes,
          IdArray rhs_nodes,
          DeviceNodeMap<IdArray> * const node_maps,
          int64_t * const count_lhs_device,
          IdArray lhs_device,
          cudaStream_t stream) {
    cudaMemsetAsync(count_lhs_device, 0, sizeof(int64_t), stream);
    node_maps->LhsHashTable(0).FillWithDuplicates(
      lhs_nodes.Ptr<IdType>(),
      lhs_nodes->shape[0],
      lhs_device.Ptr<IdType>(),
      count_lhs_device,
      stream
    );
    node_maps->RhsHashTable(0).FillWithUnique(
      rhs_nodes.Ptr<IdType>(),
      rhs_nodes->shape[0],
      stream
    );
  }

DGL_REGISTER_GLOBAL("ds.to_block._CAPI_DGLDSToBlock")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];
    const IdArray rhs_nodes = args[1];
    const bool include_rhs_in_lhs = args[2];

    //count lhs and rhs nodes
    HeteroGraphPtr graph = graph_ref.sptr();
    cudaStream_t stream = 0;
    const auto &ctx = graph->Context();
    auto device = runtime::DeviceAPI::Get(ctx);

    EdgeArray edge_array = graph->Edges(0);
    int64_t maxRHSNodes = rhs_nodes->shape[0];
    int64_t maxLHSNodes = edge_array.src->shape[0] + maxRHSNodes;

    //gather lhs_nodes
    IdArray src_nodes = NewIdArray(maxLHSNodes, ctx, sizeof(IdType) * 8);
    device->CopyDataFromTo(rhs_nodes.Ptr<IdType>(), 0, 
                           src_nodes.Ptr<IdType>(), 0,
                           sizeof(IdType) * maxRHSNodes,
                           ctx, ctx, IdType, stream);
    device->CopyDataFromTo(edge_array.src.Ptr<IdType>(), 0, 
                           src_nodes.Ptr<IdType>(), maxRHSNodes,
                           sizeof(IdType) * edge_array.src->shape[0],
                           ctx, ctx, IdType, stream);
    
    IdArray lhs_nodes = NewIdArray(maxLHSNodes, ctx, sizeof(IdType) * 8);
    int64_t *count_lhs_device = static_cast<int64_t *>(
      device->AllocWorkspace(ctx, sizeof(int64_t) * 2)
    );

    Make(src_nodes, rhs_nodes, &node_maps, count_lhs_device, lhs_nodes, stream);
    
    IdArray induced_edges = edge_array.id;

    const auto meta_graph = graph->meta_graph();
    const EdgeArray etypes = meta_graph->Edges("eid");
    const IdArray new_dst = Add(etypes.dst, 1);
    const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      2, etypes.src, new_dst
    );

    HeteroGraphPtr rel_graph;
    device->CopyDataFromTo(
      
    );

  });
  */