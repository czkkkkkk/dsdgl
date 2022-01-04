#include "./block_comm_info.h"
#include "../conn/connection.h"
#include "../utils.h"

namespace dgl {
namespace ds {


void BuildBlockCommInfo(int block_id, const std::vector<std::shared_ptr<Connection>>& conns, Coordinator* coordinator, BlockCommInfo* block_comm_info) {
  int rank = coordinator->GetRank();
  int world_size = coordinator->GetWorldSize();
  std::vector<SendDevMem*> send_dev_mem(world_size);
  std::vector<RecvDevMem*> recv_dev_mem(world_size);
  std::vector<ConnInfo> conn_infos(world_size);

  for(int offset = 0; offset < world_size; ++offset) {
    int next = (rank + offset) % world_size;
    int prev = (rank + world_size - offset) % world_size;
    ExchangeConnInfo recv_ex_info, send_ex_info;

    conns[prev]->RecvSetup(&recv_dev_mem[prev], nullptr,
                               RECV_BUFFER_SIZE, &conn_infos[prev],
                               &recv_ex_info);
    conns[next]->SendSetup(&send_dev_mem[next], nullptr,
                               RECV_BUFFER_SIZE, &conn_infos[next],
                               &send_ex_info);

    auto next_ex_info = coordinator->RingExchange(next, recv_ex_info);
    conns[next]->SendConn(&conn_infos[next], nullptr,
                              RECV_BUFFER_SIZE, &next_ex_info);

    auto prev_ex_info = coordinator->RingExchange(prev, send_ex_info);
    conns[prev]->RecvConn(&conn_infos[prev], nullptr,
                              &prev_ex_info);
  }
  DSMallocAndCopy(&block_comm_info->recv_dev_mem, recv_dev_mem);
  // DSCallocAndCopy(&info->recv_resources, recv_resources);
  DSMallocAndCopy(&block_comm_info->send_dev_mem, send_dev_mem);
  // DSCallocAndCopy(&info->send_resources, send_resources);
  DSMallocAndCopy(&block_comm_info->conn_info, conn_infos);
}


}
}