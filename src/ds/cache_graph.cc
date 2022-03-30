#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <algorithm>
#include <numeric>

#include "../c_api_common.h"
#include "./cuda/ds_kernel.h"
#include "./context.h"
#include "./utils.h"
#include "../graph/unit_graph.h"

using namespace dgl::runtime; 

namespace dgl {
namespace ds {

DGL_REGISTER_GLOBAL("ds.cache._CAPI_DGLDSCacheGraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  HeteroGraphRef hg = args[0];
  double cache_ratio = args[1];
  IdArray local_degrees = args[2];
  LOG(INFO) << "Cache graph ratio: " << cache_ratio;
  CSRMatrix mat = hg->GetCSRMatrix(0);
  IdType n_nodes = mat.indptr->shape[0] - 1;
  IdType n_edges = mat.indices->shape[0];
  IdType max_cached_edges = n_edges * (cache_ratio / 100.);

  std::vector<IdType> adj_pos_map(n_nodes, -1);
  std::vector<IdType> ids(n_nodes);
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [&local_degrees](IdType l, IdType r) {
    return local_degrees.Ptr<IdType>()[l] > local_degrees.Ptr<IdType>()[r];
  });
  
  std::vector<IdType> dev_indptr(1, 0), dev_indices, host_indptr(1, 0), host_indices;
  IdType n_cached_edges = 0;
  IdType n_cached_nodes = n_nodes;
  for(int i = 0; i < n_nodes; ++i) {
    IdType u = ids[i];
    IdType start = mat.indptr.Ptr<IdType>()[u];
    IdType end = mat.indptr.Ptr<IdType>()[u+1];
    IdType n_out_edges = end - start;
    if(n_cached_edges + n_out_edges <= max_cached_edges) {
      n_cached_edges += n_out_edges;
      dev_indptr.push_back(dev_indptr.back() + n_out_edges);
      dev_indices.insert(dev_indices.end(), mat.indices.Ptr<IdType>() + start, mat.indices.Ptr<IdType>() + end);
      adj_pos_map[u] = i;
    } else {
      n_cached_nodes = i;
      break;
    }
  }
  IdType n_uva_nodes = n_nodes - n_cached_nodes;
  for(int i = n_cached_nodes; i < n_nodes; ++i) {
    IdType u = ids[i];
    IdType start = mat.indptr.Ptr<IdType>()[u];
    IdType end = mat.indptr.Ptr<IdType>()[u+1];
    IdType n_out_edges = end - start;
    host_indptr.push_back(host_indptr.back() + n_out_edges);
    host_indices.insert(host_indices.end(), mat.indices.Ptr<IdType>() + start, mat.indices.Ptr<IdType>() + end);
    adj_pos_map[u] = ENCODE_ID(i - n_cached_nodes);
  }
  auto* context = DSContext::Global();

  int rank = context->rank;
  LOG(INFO) << "[Rank] " << rank << " Cached nodes: " << n_cached_nodes << " Cached edges: " << dev_indices.size();
  LOG(INFO) << "[Rank] " << rank << " Host nodes: " << host_indptr.size() - 1 << " host edges: " << host_indices.size();

  context->dev_graph = CSRMatrix(n_cached_nodes, n_cached_nodes, IdArray::FromVector(dev_indptr, {kDLGPU, rank}), IdArray::FromVector(dev_indices, {kDLGPU, rank}));
  context->uva_graph = CSRMatrix(n_uva_nodes, n_uva_nodes, IdArray::FromVector(host_indptr, {kDLCPU, 0}), IdArray::FromVector(host_indices, {kDLCPU, 0}));
  context->adj_pos_map = IdArray::FromVector(adj_pos_map, {kDLGPU, rank});
  Register(context->uva_graph.indptr);
  Register(context->uva_graph.indices);
  context->n_cached_nodes = n_cached_nodes;
  context->n_uva_nodes = n_uva_nodes;
  context->graph_loaded = true;
});

}
}