#include "./comm_info.h"

#include "../utils.h"

namespace dgl {
namespace ds {

void BuildCommInfo(int n_block, const std::vector<std::shared_ptr<Connection>>& conns, Coordinator* coordinator, CommInfo* info) {
  info->n_block = n_block;
  // For each block, build the block information for them
  for(int block_id = 0; block_id < n_block; ++block_id)  {
    BuildBlockCommInfo(block_id, conns, coordinator, &info->block_comm_info[block_id]);
  }
  DSMallocAndCopy(&info->dev_comm_info, info, 1);
}

}
}