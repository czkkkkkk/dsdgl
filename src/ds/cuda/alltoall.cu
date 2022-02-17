#include "./alltoall.h"

#include <dmlc/logging.h>
#include <thread>

#include "../comm/comm_info.h"
#include "../utils.h"
#include "../../runtime/cuda/cuda_common.h"
#include "../context.h"
#include "./ds_kernel.h"

using namespace dgl::runtime;

namespace dgl {
namespace ds {

// 500 MB
static constexpr int MAX_RECV_BUFFER_SIZE = 500 * 1024 * 1024;

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
  int n_bytes;
  void *input, *output;
  void *my_recvbuff, *next_recvbuff;
  int send_size, recv_size;
  WaitFlag ready, done, next_ready, prev_done;
};

template<typename T>
__device__
void _Copy(CopyArgs args) {
  static const int FETCH_BYTES = sizeof(T);
  int bid = blockIdx.x;
  if (args.tid % args.group_size == 0) {
    args.ready.post(1);
    args.next_ready.wait(1);
  }
  __syncthreads();
  int tid = args.tid;
  int buff_ptr = args.tid % args.group_size;
  int send_size = args.send_size / FETCH_BYTES;
  T* input = (T*)args.input;
  T* next_recvbuff = (T*)args.next_recvbuff;
  while(tid < send_size) {
    T val = vFetch(input + tid);
    vStore(next_recvbuff + buff_ptr, val);
    // args.next_recvbuff[buff_ptr] = args.input[tid];
    tid += args.n_threads;
    buff_ptr += args.group_size;
  }
  __threadfence_system();
  __syncthreads();
  if (args.tid % args.group_size == 0) {
    args.done.post(1);
    args.prev_done.wait(1);
  }
  __syncthreads();

  // ------- Receive -----------
  tid = args.tid;
  buff_ptr = args.tid % args.group_size;
  int recv_size = args.recv_size / FETCH_BYTES;
  T *my_recvbuff = (T*) args.my_recvbuff;
  T *output = (T*)args.output;
  while(tid < recv_size) {
    T val = vFetch(my_recvbuff + buff_ptr);
    vStore(output + tid, val);
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
  copy_args.send_size = sizeof(int64_t);
  copy_args.recv_size = sizeof(int64_t);
  copy_args.group_size = n_threads_per_conn;
  copy_args.input = send_sizes + peer_id;
  copy_args.output = recv_sizes + peer_id;
  copy_args.my_recvbuff = conn_info->my_recv_buff;
  copy_args.next_recvbuff = conn_info->next_recv_buff;
  _Copy<int64_t>(copy_args);
}

template<typename T>
__device__
void _CopyData(void* input, int64_t send_size, void* output, int64_t recv_size, int tid, int n_threads, int group_size, ConnInfo* conn_info) {
  CopyArgs copy_args(tid, n_threads, conn_info->my_ready, conn_info->my_done, conn_info->next_ready, conn_info->prev_done);
  copy_args.group_size = group_size;
  copy_args.send_size = send_size;
  copy_args.recv_size = recv_size;
  copy_args.input = input;
  copy_args.output = output;
  copy_args.my_recvbuff = conn_info->my_recv_buff;
  copy_args.next_recvbuff = conn_info->next_recv_buff;
  _Copy<T>(copy_args);
}

template<typename T>
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
    recv_offset[0] = 0;
    if(bid == gridDim.x - 1) {
      ((int64_t*)args.recv_offset)[0] = 0;
    }
    for(int i = 0; i < world_size; ++i) {
      recv_offset[i+1] = recv_offset[i] + recv_sizes[i];
      if(bid == gridDim.x - 1) {
        ((int64_t*)args.recv_offset)[i+1] = recv_offset[i+1];
      }
    }
  }
  __syncthreads();
  void* sendbuff = (T*)args.sendbuff + send_offset[peer_id] * args.n_bytes / sizeof(T);
  void* recvbuff = (T*)args.recvbuff + recv_offset[peer_id] * args.n_bytes / sizeof(T);
  int64_t send_size = (send_offset[peer_id+1] - send_offset[peer_id]) * args.n_bytes;
  int64_t recv_size = (recv_offset[peer_id+1] - recv_offset[peer_id]) * args.n_bytes;
  int global_tid = bid * args.n_threads_per_conn + local_tid;
  _CopyData<T>(sendbuff, send_size, recvbuff, recv_size, global_tid, gridDim.x * args.n_threads_per_conn, args.n_threads_per_conn, conn_info);
}

void CustomAlltoall(void* sendbuff, int64_t* send_offset, void* recvbuff, int64_t* recv_offset, int n_bytes, int align_size, CommInfo* comm_info, int rank, int world_size) {
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  AlltoallArgs args;
  args.rank = rank;
  args.world_size = world_size;
  static constexpr int MAX_THREADS = 512;
  CHECK(MAX_THREADS % world_size == 0);
  args.n_threads_per_conn = MAX_THREADS / world_size;
  int n_threads = args.n_threads_per_conn * world_size;
  args.n_bytes = n_bytes;
  args.comm_info = comm_info->dev_comm_info;
  args.sendbuff = sendbuff;
  args.send_offset = send_offset;
  args.recvbuff = recvbuff;
  args.recv_offset = recv_offset;
  dim3 grid_dim(comm_info->n_block);
  dim3 block_dim(n_threads);
  void *kargs[] = {&args};
  cudaError_t e;
  if(align_size == 4) {
    e = cudaLaunchKernel((void *)_AlltoallKernel<int>,
                                    grid_dim, block_dim, kargs, 0, thr_entry->stream);
  } else if(align_size == 8) {
    CHECK(n_bytes % 8 == 0);
    e = cudaLaunchKernel((void *)_AlltoallKernel<int64_t>,
                                    grid_dim, block_dim, kargs, 0, thr_entry->stream);
  } else {
    LOG(FATAL) << "Unsupported bytes: " << n_bytes;
  }

  CUDACHECKERR(e);
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void NCCLAllToAll(IdArray send_buffer, IdArray send_offset, IdArray recv_buffer, IdArray recv_offset, int expand_size, int rank, int world_size, ncclComm_t nccl_comm) {
  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  T* send_buffer_ptr = send_buffer.Ptr<T>();
  T* recv_buffer_ptr = recv_buffer.Ptr<T>();
  int type_bytes = sizeof(T);
  IdType* send_offset_ptr = send_offset.Ptr<IdType>();
  IdType* recv_offset_ptr = recv_offset.Ptr<IdType>();
  CUDACHECK(cudaMemcpyAsync(recv_buffer_ptr + recv_offset_ptr[rank] * expand_size, 
                       send_buffer_ptr + send_offset_ptr[rank] * expand_size, 
                       (send_offset_ptr[rank + 1] - send_offset_ptr[rank]) * expand_size * type_bytes, cudaMemcpyDeviceToDevice, stream));
  ncclGroupStart();
  for(int r = 0; r < world_size; ++r) {
    if(r != rank) {
      IdType send_size = (send_offset_ptr[r+1] - send_offset_ptr[r]) * expand_size;
      IdType send_ptr = send_offset_ptr[r] * expand_size;
      IdType recv_size = (recv_offset_ptr[r+1] - recv_offset_ptr[r]) * expand_size;
      IdType recv_ptr = recv_offset_ptr[r] * expand_size;
      ncclSend(send_buffer_ptr + send_ptr, send_size, NCCL_DATA_TYPE, r, nccl_comm, stream);
      ncclRecv(recv_buffer_ptr + recv_ptr, recv_size, NCCL_DATA_TYPE, r, nccl_comm, stream);
    }
  }
  ncclGroupEnd();
}

__global__ 
void _DiffKernel(IdType* out, IdType* in, int world_size) {
  int tid = threadIdx.x;
  if(tid < world_size) {
    out[tid] = in[tid + 1] - in[tid];
  }
}

std::pair<IdArray, IdArray> Alltoall(IdArray input, IdArray send_offset, int expand_size, int rank, int world_size) {
  if(!GetEnvParam("USE_NCCL", 1)) {
    auto stream = CUDAThreadEntry::ThreadLocal()->stream;
    auto* ds_context = DSContext::Global();
    auto dgl_context = input->ctx;
    auto recvbuff = IdArray::Empty({MAX_RECV_BUFFER_SIZE / (input->dtype.bits / 8)}, input->dtype, dgl_context);
    IdArray recv_offset = IdArray::Empty({world_size + 1}, send_offset->dtype, dgl_context);
    CustomAlltoall(input.Ptr<IdType>(), send_offset.Ptr<IdType>(), recvbuff.Ptr<IdType>(), recv_offset.Ptr<IdType>(), input->dtype.bits / 8 * expand_size, input->dtype.bits / 8, &ds_context->comm_info, rank, world_size);

    auto host_recv_offset = recv_offset.CopyTo({kDLCPU, 0}, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
    int64_t* host_recv_offset_ptr = host_recv_offset.Ptr<int64_t>();
    CHECK_LE(host_recv_offset_ptr[world_size] * expand_size * input->dtype.bits / 8, MAX_RECV_BUFFER_SIZE);
    recvbuff = recvbuff.CreateView({(signed long) host_recv_offset_ptr[world_size] * expand_size}, input->dtype);
    return {recvbuff, recv_offset};
  } else {
    // NCCL
    CHECK(send_offset->dtype.bits == 64);
    auto stream = CUDAThreadEntry::ThreadLocal()->stream;
    auto dgl_context = input->ctx;
    auto host_dgl_context = DLContext{kDLCPU, 0};
    auto nccl_comm = DSContext::Global()->nccl_comm;
    IdArray send_sizes = IdArray::Empty({world_size}, send_offset->dtype, dgl_context);
    _DiffKernel<<<1, 32, 0, stream>>>(send_sizes.Ptr<IdType>(), send_offset.Ptr<IdType>(), world_size);
    CUDACHECK(cudaGetLastError());
    IdArray recv_sizes = IdArray::Empty({world_size}, send_offset->dtype, dgl_context);
    IdArray range_seq = Range(0, world_size + 1, 64, host_dgl_context);
    NCCLAllToAll<int64_t, ncclInt64>(send_sizes, range_seq, recv_sizes, range_seq, 1, rank, world_size, nccl_comm);
    auto host_send_offset = send_offset.CopyTo(host_dgl_context, stream);

    auto recv_offset = CumSum(recv_sizes, true);
    IdArray host_recv_offset = recv_offset.CopyTo(host_dgl_context, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
    auto* host_recv_offset_ptr = host_recv_offset.Ptr<IdType>();
    int n_recv = host_recv_offset_ptr[world_size] * expand_size;
    auto recvbuff = IdArray::Empty({n_recv}, input->dtype, dgl_context);
    if(input->dtype.bits == 32) {
      NCCLAllToAll<int, ncclInt32>(input, host_send_offset, recvbuff, host_recv_offset, expand_size, rank, world_size, nccl_comm);
    } else {
      NCCLAllToAll<int64_t, ncclInt64>(input, host_send_offset, recvbuff, host_recv_offset, expand_size, rank, world_size, nccl_comm);
    }
    return {recvbuff, recv_offset};
  }
}

}
}
