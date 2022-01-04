#include "./p2p_connection.h"

#include "../comm/comm_info.h"
#include "../utils.h"

namespace dgl {
namespace ds {

void P2pConnection::SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                      int buffer_size, ConnInfo* conn_info,
                      ExchangeConnInfo* ex_info) {
  DSCudaMalloc(send_dev_mem);
  conn_info->my_done = &(*send_dev_mem)->done;

  P2pExchangeConnInfo p2p_info;
  if(!my_info_.SameDevice(peer_info_)) {
    cudaIpcGetMemHandle(&p2p_info.dev_ipc, (void*)(*send_dev_mem));
    static_assert(sizeof(P2pExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                  "P2P exchange connection info too large");
  } else {
    p2p_info.ptr = *send_dev_mem;
  }
  memcpy(ex_info, &p2p_info, sizeof(P2pExchangeConnInfo));
}

void P2pConnection::RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                int buffer_size, ConnInfo* conn_info,
                ExchangeConnInfo* ex_info) {
  int dev_mem_size = offsetof(RecvDevMem, buff) + buffer_size;
  DSCudaMalloc((char**)recv_dev_mem, dev_mem_size);

  conn_info->my_ready = &(*recv_dev_mem)->ready;
  conn_info->my_recv_buff = (*recv_dev_mem)->buff;

  P2pExchangeConnInfo p2p_info;
  if(!my_info_.SameDevice(peer_info_)) {
    cudaIpcGetMemHandle(&p2p_info.dev_ipc, (void*)(*recv_dev_mem));
    static_assert(sizeof(P2pExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                  "P2P exchange connection info too large");
  } else {
    p2p_info.ptr = *recv_dev_mem;
  }
  memcpy(ex_info, &p2p_info, sizeof(P2pExchangeConnInfo));
}

void P2pConnection::SendConn(ConnInfo* conn_info, void* send_resources, int buffer_size,
              ExchangeConnInfo* peer_ex_info)  {
  P2pExchangeConnInfo* peer_info = (P2pExchangeConnInfo*)peer_ex_info;
  RecvDevMem* ptr;
  if(!my_info_.SameDevice(peer_info_)) {
    CUDACHECK(cudaIpcOpenMemHandle((void**)&ptr, peer_info->dev_ipc,
                                  cudaIpcMemLazyEnablePeerAccess));
  } else {
    ptr = (RecvDevMem*)peer_info->ptr;
  }
  conn_info->next_ready = &ptr->ready;
  conn_info->next_recv_buff = &ptr->buff;
}

void P2pConnection::RecvConn(ConnInfo* conn_info, void* recv_resources,
              ExchangeConnInfo* peer_ex_info) {
  P2pExchangeConnInfo* peer_info = (P2pExchangeConnInfo*)peer_ex_info;
  SendDevMem* ptr;
  if(!my_info_.SameDevice(peer_info_)) {
    CUDACHECK(cudaIpcOpenMemHandle((void**)&ptr, peer_info->dev_ipc,
                                  cudaIpcMemLazyEnablePeerAccess));
  } else {
    ptr = (SendDevMem*)peer_info->ptr;
  }
  conn_info->prev_done = &ptr->done;
}

}
}