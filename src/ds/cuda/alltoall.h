#ifndef DGL_DS_CUDA_ALLTOALL_H_
#define DGL_DS_CUDA_ALLTOALL_H_

#include "../comm/comm_info.h"

#include <dgl/array.h>

namespace dgl {
namespace ds {

struct AlltoallArgs {
  int rank, world_size;
  int n_bytes;
  CommInfo* comm_info;
  int n_threads_per_conn;
  void *sendbuff, *send_offset;
  void *recvbuff, *recv_offset;
};

/**
 * @brief Perform an alltoall operation. 
 * 
 * @param senfbuff      sendbuff on GPU
 * @param senf_offset   send_offset on GPU, which counts the number of bytes (must be 64 bits)
 * @param recvbuff      recvbuff on GPU
 * @param recv_offset   recv_offset on GPU, which counts the number of bytes (must be 64 bits)
 * @param n_bytes       Size of each element in bytes
 * @param comm_info     Communication information
 * @param rank        
 * @param world_size
 */
void CustomizedAlltoall(void* sendbuff, int64_t* send_offset, void* recvbuff, int64_t* recv_offset, int n_bytes, CommInfo *comm_info, int rank, int world_size);


/**
 * @brief Unified alltoall communication.
 * 
 * @param input input on GPU
 * @param send_offset send_offset on GPU
 * @param rank
 * @param world_size
 * 
 * @return Tuple of (received buff, recv_sizes, recv_offset)
 */
std::pair<IdArray, IdArray> Alltoall(IdArray input, IdArray send_offset, int expand_size, int rank, int world_size);

}
}

#endif
