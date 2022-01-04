#ifndef DGL_DS_CUDA_ALLTOALL_H_
#define DGL_DS_CUDA_ALLTOALL_H_

#include "../comm/comm_info.h"

namespace dgl {
namespace ds {

struct AlltoallArgs {
  int rank, world_size;
  CommInfo* comm_info;
  int n_threads_per_conn;
  void *sendbuff, *send_offset;
  void *recvbuff, *recv_offset;
};

void Alltoall(void* sendbuff, void* send_offset, void* recvbuff, void* recv_offset, CommInfo *comm_info, int rank, int world_size);

}
}

#endif