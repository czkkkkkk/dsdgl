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

HeteroSubgraph SampleNeighbors(
    const HeteroGraphPtr hg,
    const IdArray& nodes,
    int fanout,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace,
    ) {
      // std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
      // std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
      // dgl_type_t etype = 0;
      // auto pair = hg->meta_graph()->FindEdge(etype);
      // const dgl_type_t src_vtype = pair.first;
      // const dgl_type_t dst_vtype = pair.second;
      // const IdArray nodes_ntype = nodes[(dir == EdgeDir::kOut)? src_vtype : dst_vtype];
      // const int64_t num_nodes = nodes_ntype->shape[0];

      // auto req_fmt = (dir == EdgeDir::kOut)? CSR_CODE : CSC_CODE;
      // auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      // COOMatrix sampled_coo;

      // switch (avail_fmt) {
      //   case SparseFormat::kCOO:
      //     if (dir == EdgeDir::kIn) {
      //       sampled_coo = aten::COOTranspose(aten::COORowWiseSampling(
      //         aten::COOTranspose(hg->GetCOOMatrix(etype)),
      //         nodes_ntype, fanouts[etype], prob[etype], replace));
      //     } else {
      //       sampled_coo = aten::COORowWiseSampling(
      //         hg->GetCOOMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
      //     }
      //     break;
      //   case SparseFormat::kCSR:
      //     CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
      //     sampled_coo = aten::CSRRowWiseSampling(
      //       hg->GetCSRMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
      //     break;
      //   case SparseFormat::kCSC:
      //     CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
      //     sampled_coo = aten::CSRRowWiseSampling(
      //       hg->GetCSCMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
      //     sampled_coo = aten::COOTranspose(sampled_coo);
      //     break;
      //   default:
      //     LOG(FATAL) << "Unsupported sparse format.";
      // }
      // subrels[etype] = UnitGraph::CreateFromCOO(
      //   hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows, sampled_coo.num_cols,
      //   sampled_coo.row, sampled_coo.col);
      // induced_edges[etype] = sampled_coo.data;

      HeteroSubgraph ret;
      // ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
      // ret.induced_vertices.resize(hg->NumVertexTypes());
      // ret.induced_edges = std::move(induced_edges);
      return ret;
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
    uint64 *d_seeds = static_cast<uint64*>(nodes->data);
    uint64 *h_device_col_ptr = new uint64[num_devices + 1]; 
    uint64 *h_device_col_cnt = new uint64[num_devices];
    uint64 *d_device_col_cnt = nullptr;
    Cluster(num_devices, d_device_vids, num_seeds, d_seeds, fanout, 
            h_device_col_ptr, h_device_col_cnt, &d_device_col_cnt);
    
    uint64 num_frontier;
    int *d_frontier;
    int h_device_offset[num_devices + 1];
    Shuffle(num_devices, h_device_col_ptr, h_device_col_cnt, d_device_col_cnt,
            d_seeds, num_frontier, h_device_offset, &d_frontier,
            context->rank, context->nccl_comm);

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    // *subg = ds::SampleNeighbors(
    //     hg.sptr(), min_ids, nodes, fanout, dir_str, prob, replace, context);

    *rv = HeteroSubgraphRef(subg);
  });

}
}