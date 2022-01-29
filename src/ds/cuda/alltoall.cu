#include "./alltoall.h"

#include <dmlc/logging.h>

#include "../comm/comm_info.h"
#include "../utils.h"

namespace dgl {
namespace ds {

__device__
void sleep(int clock_count) {
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock() - start_clock;
  }

}

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

class WaitFlag {
  volatile uint64_t *const flag;

 public:
  __host__ __device__ __forceinline__ WaitFlag(volatile uint64_t *const flag)
      : flag(flag) {
      }
  __device__ uint64_t get_flag() { return *flag; }
  __device__ __forceinline__ void unset() { post(FLAG_UNUSED); }
  __device__ __forceinline__ void wait_unset() { wait(FLAG_UNUSED); }
  
  __device__ __forceinline__ void wait(uint64_t val) {
    /*SPIN*/
    while ((*flag) != val) {
    }
  }
  __device__ __forceinline__ void post(uint64_t val) { *flag = val; }
  const static uint64_t FLAG_UNUSED = ~0ull >> 1;
};

struct CopyArgs {
  __host__ __device__ __forceinline__ CopyArgs(int tid, int n_threads,
                                               uint64_t *ready, uint64_t *done,
                                               uint64_t *next_ready,
                                               uint64_t *prev_done)
      : tid(tid),
        n_threads(n_threads),
        ready(ready),
        done(done),
        next_ready(next_ready),
        prev_done(prev_done) {}
  int tid, n_threads, group_size;
  // Pack 64 bits currently
  int64_t *input, *output;
  int64_t *my_recvbuff, *next_recvbuff;
  int send_size, recv_size;
  WaitFlag ready, done, next_ready, prev_done;
};

template <typename T=int64_t>
__device__
void _Copy(CopyArgs args) {
  if (args.tid % args.group_size == 0) {
    args.ready.post(1);
    args.next_ready.wait(1);
  }
  __syncthreads();
  int tid = args.tid;
  int buff_ptr = args.tid % args.group_size;
  while(tid < args.send_size) {
    T val = vFetch(((T*)args.input) + tid);
    vStore(((T*)args.next_recvbuff + buff_ptr), val);
    // args.next_recvbuff[buff_ptr] = args.input[tid];
    tid += args.n_threads;
    buff_ptr += args.group_size;
  }
  __syncthreads();
  if (args.tid % args.group_size == 0) {
    args.done.post(1);
    args.prev_done.wait(1);
  }
  __syncthreads();
  tid = args.tid;
  buff_ptr = args.tid % args.group_size;
  while(tid < args.recv_size) {
    T val = vFetch(((T*)args.my_recvbuff) + buff_ptr);
    vStore(((T*)args.output) + tid, val);
    // args.output[tid] = args.my_recvbuff[buff_ptr]; 
    tid += args.n_threads;
    buff_ptr += args.group_size;
  }
  __syncthreads();
  if (args.tid % args.group_size == 0) {
    args.ready.unset();
    args.next_ready.wait_unset();
    args.done.unset();
    args.prev_done.wait_unset();
  }
  __syncthreads();
}

__device__
void _CopySendSize(int64_t* send_sizes, int64_t* recv_sizes, int peer_id, int local_tid, int n_threads_per_conn, ConnInfo* conn_info) {
  CopyArgs copy_args(local_tid, n_threads_per_conn, conn_info->my_ready, conn_info->my_done, conn_info->next_ready, conn_info->prev_done);
  copy_args.send_size = 1;
  copy_args.recv_size = 1;
  copy_args.group_size = n_threads_per_conn;
  copy_args.input = send_sizes + peer_id;
  copy_args.output = recv_sizes + peer_id;
  copy_args.my_recvbuff = (int64_t*) conn_info->my_recv_buff;
  copy_args.next_recvbuff = (int64_t*) conn_info->next_recv_buff;
  _Copy(copy_args);
}

template <typename T>
__device__
void _CopyData(T* input, int64_t send_size, T* output, int64_t recv_size, int tid, int n_threads, int group_size, ConnInfo* conn_info) {
  CopyArgs copy_args(tid, n_threads, conn_info->my_ready, conn_info->my_done, conn_info->next_ready, conn_info->prev_done);
  copy_args.group_size = group_size;
  copy_args.send_size = send_size;
  copy_args.recv_size = recv_size;
  copy_args.input = (int64_t*) input;
  copy_args.output = (int64_t*) output;
  copy_args.my_recvbuff = (int64_t*) conn_info->my_recv_buff;
  copy_args.next_recvbuff = (int64_t*) conn_info->next_recv_buff;
  if (sizeof(T) == 8) {
    _Copy<int64_t>(copy_args);
  } else {
    _Copy<int32_t>(copy_args);
  }
}

template <typename T>
__global__
void _AlltoallKernel(AlltoallArgs args) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int world_size = args.world_size;
  int rank = args.rank;
  int peer_id = tid / args.n_threads_per_conn;
  int local_tid = tid % args.n_threads_per_conn;
  ConnInfo* conn_info = args.comm_info->block_comm_info[bid].conn_info + peer_id;

  __shared__ int64_t send_sizes[8], recv_sizes[8], recv_offset[9];
  int64_t* send_offset = (int64_t*)args.send_offset;
  if(tid < args.world_size) {
    send_sizes[tid] = send_offset[tid + 1] - send_offset[tid];
  }
  __syncthreads();

  _CopySendSize(send_sizes, recv_sizes, peer_id, local_tid, args.n_threads_per_conn, conn_info);
  if(tid == 0) {
    ((int64_t*)args.recv_offset)[0] = recv_offset[0] = 0;
    for(int i = 0; i < world_size; ++i) {
      ((int64_t*)args.recv_offset)[i+1] = recv_offset[i+1] = recv_offset[i] + recv_sizes[i];
    }
  }
  __syncthreads();
  T* sendbuff = ((T*) args.sendbuff) + send_offset[peer_id];
  T* recvbuff = ((T*) args.recvbuff) + recv_offset[peer_id];
  int64_t send_size = send_offset[peer_id+1] - send_offset[peer_id];
  int64_t recv_size = recv_offset[peer_id+1] - recv_offset[peer_id];
  int global_tid = bid * args.n_threads_per_conn + local_tid;
  _CopyData(sendbuff, send_size, recvbuff, recv_size, global_tid, gridDim.x * args.n_threads_per_conn, args.n_threads_per_conn, conn_info);
}

void Alltoall(void* sendbuff, void* send_offset, void* recvbuff, void* recv_offset, CommInfo* comm_info, int rank, int world_size, int type_bytes) {
  AlltoallArgs args;
  args.rank = rank;
  args.world_size = world_size;
  args.n_threads_per_conn = 64;
  int n_threads = args.n_threads_per_conn * world_size;
  args.comm_info = comm_info->dev_comm_info;
  args.sendbuff = sendbuff;
  args.send_offset = send_offset;
  args.recvbuff = recvbuff;
  args.recv_offset = recv_offset;
  dim3 grid_dim(comm_info->n_block);
  dim3 block_dim(n_threads);
  void *kargs[] = {&args};
  cudaError_t e = cudaSuccess;
  if (type_bytes == 8) {
    e = cudaLaunchKernel((void *)(_AlltoallKernel<int64_t>), grid_dim, block_dim, kargs, 0, 0);
    //e = _AlltoallKernel<int64_t><<<grid_dim, block_dim>>>(kargs);
  } else {
    e = cudaLaunchKernel((void *)(_AlltoallKernel<int32_t>), grid_dim, block_dim, kargs, 0, 0);
    //e = _AlltoallKernel<int32_t><<<grid_dim, block_dim>>>(kargs);
  }
  CUDACHECKERR(e);
  CUDACHECK(cudaStreamSynchronize(0));
}

}
}