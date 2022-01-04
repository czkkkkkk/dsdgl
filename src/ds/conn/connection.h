#ifndef DGL_DS_COMM_CONNECTION_H
#define DGL_DS_COMM_CONNECTION_H

#include <memory>

#include "../coordinator.h"

#define MAX_EXCHANGE_CONN_INFO_SIZE 256

namespace dgl {
namespace ds {

enum ConnType {P2P};

struct SendDevMem;
struct RecvDevMem;
struct ConnInfo;

struct ExchangeConnInfo {
  char info[MAX_EXCHANGE_CONN_INFO_SIZE];
};

class Connection {
 public:
  Connection(ProcInfo my_info, ProcInfo peer_info): my_info_(my_info), peer_info_(peer_info) {}

  // Build a connection
  static std::shared_ptr<Connection> GetConnection(ProcInfo r1, ProcInfo r2);

  virtual void SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                        int buffer_size, ConnInfo* conn_info,
                        ExchangeConnInfo* ex_info) = 0;
  virtual void RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                         int buffer_size, ConnInfo* conn_info,
                         ExchangeConnInfo* ex_info) = 0;

  virtual void SendConn(ConnInfo* conn_info, void* send_resources,
                        int buffer_size, ExchangeConnInfo* peer_ex_info) = 0;
  virtual void RecvConn(ConnInfo* conn_info, void* recv_resources,
                        ExchangeConnInfo* peer_ex_info) = 0;

  ProcInfo GetMyInfo() const { return my_info_; }
  ProcInfo GetPeerInfo() const { return peer_info_; }
 protected:
  ProcInfo my_info_, peer_info_;
};

}
}

#endif