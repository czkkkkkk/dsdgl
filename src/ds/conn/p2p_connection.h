#ifndef DGL_DS_COMM_P2P_CONNECTION_H_
#define DGL_DS_COMM_P2P_CONNECTION_H_

#include "./connection.h"

#include <cuda_runtime.h>

namespace dgl {
namespace ds {

struct P2pExchangeConnInfo {
  cudaIpcMemHandle_t dev_ipc;
  void* ptr;
};

class P2pConnection: public Connection {
 public:
  P2pConnection(ProcInfo my_info, ProcInfo peer_info): Connection(my_info, peer_info) {}
  void SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                        int buffer_size, ConnInfo* conn_info,
                        ExchangeConnInfo* ex_info);

  void RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                int buffer_size, ConnInfo* conn_info,
                ExchangeConnInfo* ex_info) override;

  void SendConn(ConnInfo* conn_info, void* send_resources, int buffer_size,
                ExchangeConnInfo* peer_ex_info) override;

  void RecvConn(ConnInfo* conn_info, void* recv_resources,
                ExchangeConnInfo* peer_ex_info) override;
};

}
}
#endif