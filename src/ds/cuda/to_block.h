#ifndef DGL_DS_TOBLOCK_H_
#define DGL_DS_TOBLOCK_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <vector>
#include <tuple>

namespace dgl {

namespace ds {
  std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray> >
  ToBlock(HeteroGraphPtr graph,
          const std::vector<IdArray> &rhs_nodes,
          bool include_rhs_in_lhs);

}

}

#endif