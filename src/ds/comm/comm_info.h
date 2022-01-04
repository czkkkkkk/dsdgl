#ifndef DGL_DS_COMM_COMM_INFO_H_
#define DGL_DS_COMM_COMM_INFO_H_

#include "./block_comm_info.h"
#include "../coordinator.h"
#include "../conn/connection.h"

namespace dgl {
namespace ds {

#define MAX_COMM_BLOCK 32

/**
 * @brief CommInfo stores communication information, including send/recv buffer pointers of other ranks.
 *        It is used as an argument passing to the communication kernel.
 */
struct CommInfo {
  BlockCommInfo block_comm_info[MAX_COMM_BLOCK];
  CommInfo* dev_comm_info;
  int n_block;
};

void BuildCommInfo(int n_block, const std::vector<std::shared_ptr<Connection>>& conns, Coordinator* coordinator, CommInfo* info);

}
}

#endif