#include "./alltoall.h"

#include <dmlc/logging.h>
#include <thread>
#define _CG_ABI_EXPERIMENTAL // enable experimental API
#include <cooperative_groups.h>

#include "../comm/comm_info.h"
#include "../utils.h"
#include "../../runtime/cuda/cuda_common.h"
#include "../context.h"
#include "./ds_kernel.h"
#include "../schedule.h"

using namespace dgl::runtime;
using namespace cooperative_groups;


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
  __device__ __forceinline__ void init() { post(FLAG_INIT); }
  __device__ __forceinline__ void wait_init() { wait(FLAG_INIT); }
  __device__ __forceinline__ void unset() { post(FLAG_UNUSED); }
  __device__ __forceinline__ void wait_unset() { wait(FLAG_UNUSED); }
  
  __device__ __forceinline__ void wait(uint64_t val) {
    /*SPIN*/
    while ((*flag) != val) {
    }
  }
  __device__ __forceinline__ void post(uint64_t val) { *flag = val; }
  static constexpr uint64_t FLAG_INIT = ~0ull >> 1;
  static constexpr uint64_t FLAG_UNUSED = (~0ull >> 1) - 1;
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
  
  int rank, peer_id;
};

#define DIVUP(x, y) ((x)+(y)-1)/(y)

template<typename T, int GroupSize>
__device__
void _Copy(CopyArgs args) {
  __shared__ experimental::block_tile_memory<8> shared;
  thread_block thb = experimental::this_thread_block(shared);
  auto thread_group = experimental::tiled_partition<GroupSize>(thb);

  constexpr int FETCH_BYTES = sizeof(T);
  constexpr int n_per_substage = RECV_BUFFER_SIZE / FETCH_BYTES;
  int send_size = args.send_size / FETCH_BYTES;
  int recv_size = args.recv_size / FETCH_BYTES;
  int send_substages = DIVUP(send_size, n_per_substage);
  int recv_substages = DIVUP(recv_size, n_per_substage);
  int n_substages = send_substages > recv_substages? send_substages:recv_substages;
  int bid = blockIdx.x;
  int local_tid = args.tid % args.group_size;
  if (local_tid == 0) {
    args.done.init();
    args.prev_done.wait_init();
    args.ready.init();
    args.next_ready.wait_init();
  }
  thread_group.sync();
  int send_ptr = args.tid;
  T* input = (T*)args.input;
  T* next_recvbuff = (T*)args.next_recvbuff;

  int recv_ptr = args.tid;
  T *my_recvbuff = (T*) args.my_recvbuff;
  T *output = (T*)args.output;

  for(int substage = 0; substage < n_substages; ++substage){
    int send_buff_ptr = local_tid;
    while(send_ptr < send_size && send_buff_ptr < n_per_substage) {
      T val = vFetch(input + send_ptr);
      vStore(next_recvbuff + send_buff_ptr, val);
      send_ptr += args.n_threads;
      send_buff_ptr += args.group_size;
    }
    thread_group.sync();
    __threadfence_system();
    if (local_tid == 0) {
      args.done.post(substage);
      args.prev_done.wait(substage);
    }
    thread_group.sync();
    int recv_buff_ptr = local_tid;
    while(recv_ptr < recv_size && recv_buff_ptr < n_per_substage) {
      T val = vFetch(my_recvbuff + recv_buff_ptr);
      vStore(output + recv_ptr, val);
      recv_ptr += args.n_threads;
      recv_buff_ptr += args.group_size;
    }
    thread_group.sync();
    if (local_tid == 0) {
      args.ready.post(substage);
      args.next_ready.wait(substage);
    }
    thread_group.sync();
  }
  if (local_tid == 0) {
    args.done.unset();
    args.prev_done.wait_unset();
    args.ready.unset();
    args.next_ready.wait_unset();
  }
  thread_group.sync();
}

// Deprecated
template<int GroupSize>
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
  _Copy<int64_t, GroupSize>(copy_args);
}

template<typename T, int GroupSize>
__device__
void _CopyData(void* input, int64_t send_size, void* output, int64_t recv_size, int tid, int n_threads, int group_size, ConnInfo* conn_info, int rank, int peer_id) {
  CopyArgs copy_args(tid, n_threads, conn_info->my_ready, conn_info->my_done, conn_info->next_ready, conn_info->prev_done);
  copy_args.group_size = group_size;
  copy_args.send_size = send_size;
  copy_args.recv_size = recv_size;
  copy_args.input = input;
  copy_args.output = output;
  copy_args.my_recvbuff = conn_info->my_recv_buff;
  copy_args.next_recvbuff = conn_info->next_recv_buff;
  copy_args.rank = rank;
  copy_args.peer_id = peer_id;
  _Copy<T, GroupSize>(copy_args);
}

__device__ 
uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

template<typename T, bool exclusive, int GroupSize>
__global__
void _AlltoallKernel(AlltoallArgs args) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    atomicAdd(args.cuda_launch_lock, -1);
  }
  int world_size = args.world_size;
  int rank = args.rank;
  int peer_id = tid / args.n_threads_per_conn;
  if(exclusive && peer_id >= rank) {
    peer_id++;
  }
  int local_tid = tid % args.n_threads_per_conn;
  ConnInfo* conn_info = args.comm_info->block_comm_info[bid].conn_info + peer_id;

  __shared__ IdType send_offset[9], recv_offset[9];
  if(tid <= args.world_size) {
    send_offset[tid] = args.send_offset == nullptr? tid: args.send_offset[tid];
    recv_offset[tid] = args.recv_offset == nullptr? tid: args.recv_offset[tid];
  }
  __syncthreads();

  void* sendbuff = (T*)args.sendbuff + send_offset[peer_id] * args.n_bytes / sizeof(T);
  void* recvbuff = (T*)args.recvbuff + recv_offset[peer_id] * args.n_bytes / sizeof(T);
  int64_t send_size = (send_offset[peer_id+1] - send_offset[peer_id]) * args.n_bytes;
  int64_t recv_size = (recv_offset[peer_id+1] - recv_offset[peer_id]) * args.n_bytes;
  int global_tid = bid * args.n_threads_per_conn + local_tid;
  _CopyData<T, GroupSize>(sendbuff, send_size, recvbuff, recv_size, global_tid, gridDim.x * args.n_threads_per_conn, args.n_threads_per_conn, conn_info, rank, peer_id);
}

__global__ 
void _DiffKernel(IdType* out, IdType* in, int size) {
  int tid = threadIdx.x;
  if(tid < size) {
    out[tid] = in[tid + 1] - in[tid];
  }
}

IdArray Diff(IdArray prefix_sum) {
  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  int size = prefix_sum->shape[0] - 1;
  IdArray ret = IdArray::Empty({size}, prefix_sum->dtype, prefix_sum->ctx);
  _DiffKernel<<<1, 32, 0, stream>>>(ret.Ptr<IdType>(), prefix_sum.Ptr<IdType>(), size);
  CUDACHECK(cudaGetLastError());
  return ret;
}

#define ALLTOALL_SWITCH_ALIGN_SIZE(val, AlignType, ...) do {                 \
  if ((val) == 4) {                                             \
    using AlignType = int;                                      \
    {__VA_ARGS__}                                               \
  }                                                             \
  else if((val) == 8) {                                         \
    using AlignType = int64_t;                                  \
    {__VA_ARGS__}                                               \
  } else {                                                      \
    LOG(FATAL) << "Align size error";                           \
  }                                                             \
} while (0)

#define ALLTOALL_SWITCH_GROUP_SIZE(val, GroupSize, ...) do {                 \
  if ((val) == 16) {                                            \
    constexpr int GroupSize = 16;                               \
    {__VA_ARGS__}                                               \
  }                                                             \
  else if((val) == 64) {                                       \
    constexpr int GroupSize = 64;                              \
    {__VA_ARGS__}                                               \
  }                                                             \
  else if((val) == 128) {                                       \
    constexpr int GroupSize = 128;                              \
    {__VA_ARGS__}                                               \
  }                                                             \
  else if((val) == 256) {                                       \
    constexpr int GroupSize = 256;                              \
    {__VA_ARGS__}                                               \
  }                                                             \
  else if((val) == 512) {                                       \
    constexpr int GroupSize = 512;                              \
    {__VA_ARGS__}                                               \
  } else {                                                      \
    LOG(FATAL) << "Unsupport alltoall group size" << (val);     \
  }                                                             \
} while (0)

void CustomAlltoall(void* sendbuff, int64_t* send_offset, void* recvbuff, int64_t* recv_offset, int n_bytes, int align_size, CommInfo* comm_info, int rank, int world_size, int *cuda_launch_lock) {
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  AlltoallArgs args;
  args.rank = rank;
  args.world_size = world_size;
  static constexpr int MAX_THREADS = 512;
  CHECK(MAX_THREADS % world_size == 0);
  args.n_threads_per_conn = MAX_THREADS / world_size;
  int n_threads = args.n_threads_per_conn * (world_size - 1);
  args.n_bytes = n_bytes;
  args.comm_info = comm_info->dev_comm_info;
  args.sendbuff = sendbuff;
  args.send_offset = send_offset;
  args.recvbuff = recvbuff;
  args.recv_offset = recv_offset;
  *cuda_launch_lock = comm_info->n_block;
  args.cuda_launch_lock = cuda_launch_lock;
  dim3 grid_dim(comm_info->n_block);
  dim3 block_dim(n_threads);
  void *kargs[] = {&args};
  ALLTOALL_SWITCH_ALIGN_SIZE(align_size, AlignType, {
    ALLTOALL_SWITCH_GROUP_SIZE(args.n_threads_per_conn, GroupSize, {
      CUDACHECK(cudaLaunchKernel((void *)_AlltoallKernel<AlignType, true, GroupSize>,
                                      grid_dim, block_dim, kargs, 0, thr_entry->stream));
    });
  });

}

IdArray ExchangeSendSizes(IdArray send_offset, CommInfo* comm_info, int rank, int world_size, int *cuda_launch_lock) {
  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  auto send_sizes = Diff(send_offset);
  IdArray recv_sizes = IdArray::Empty({world_size}, send_offset->dtype, send_offset->ctx);

  AlltoallArgs args;
  args.rank = rank;
  args.world_size = world_size;
  args.n_threads_per_conn = 16;
  int n_threads = args.n_threads_per_conn * world_size;
  args.n_bytes = sizeof(IdType);
  args.comm_info = comm_info->dev_comm_info;
  args.sendbuff = send_sizes.Ptr<IdType>();
  args.send_offset = nullptr;
  args.recvbuff = recv_sizes.Ptr<IdType>();
  args.recv_offset = nullptr;
  dim3 grid_dim(1);
  dim3 block_dim(n_threads);
  *cuda_launch_lock = 1;
  args.cuda_launch_lock = cuda_launch_lock;
  void *kargs[] = {&args};
  ALLTOALL_SWITCH_ALIGN_SIZE(sizeof(IdType), AlignType, {
    ALLTOALL_SWITCH_GROUP_SIZE(args.n_threads_per_conn, GroupSize, {
      CUDACHECK(cudaLaunchKernel((void *)_AlltoallKernel<AlignType, false, GroupSize>,
                                      grid_dim, block_dim, kargs, 0, stream));
    });
  });
  auto recv_offset = CumSum(recv_sizes, true);
  return recv_offset;
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void NCCLAllToAll(IdArray send_buffer, IdArray send_offset, IdArray recv_buffer, IdArray recv_offset, int expand_size, int rank, int world_size, ncclComm_t nccl_comm) {
  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  auto data_copy_stream = CUDAThreadEntry::ThreadLocal()->data_copy_stream;
  T* send_buffer_ptr = send_buffer.Ptr<T>();
  T* recv_buffer_ptr = recv_buffer.Ptr<T>();
  int type_bytes = sizeof(T);
  IdType* send_offset_ptr = send_offset.Ptr<IdType>();
  IdType* recv_offset_ptr = recv_offset.Ptr<IdType>();
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaMemcpyAsync(recv_buffer_ptr + recv_offset_ptr[rank] * expand_size, 
                       send_buffer_ptr + send_offset_ptr[rank] * expand_size, 
                       (send_offset_ptr[rank + 1] - send_offset_ptr[rank]) * expand_size * type_bytes, cudaMemcpyDeviceToDevice, data_copy_stream));
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

std::pair<IdArray, IdArray> Alltoall(IdArray input, IdArray send_offset, int expand_size, int rank, int world_size) {
  auto* scheduler = Scheduler::Global();
  if(!GetEnvParam("USE_NCCL", 1)) {
    auto stream = CUDAThreadEntry::ThreadLocal()->stream;
    auto data_copy_stream = CUDAThreadEntry::ThreadLocal()->data_copy_stream;
    auto *ds_context = DSContext::Global();
    auto dgl_context = input->ctx;
    int type_bytes = input->dtype.bits / 8;
    int *cuda_launch_lock = &(CUDAThreadEntry::ThreadLocal()->cuda_launch_lock);
    int thread_id = CUDAThreadEntry::ThreadLocal()->thread_id;

    // NOTE: to guarantee the send_offset is ready
    CUDACHECK(cudaStreamSynchronize(stream));
    auto host_send_offset = send_offset.CopyTo({kDLCPU, 0}, data_copy_stream);

    CommInfo *comm_info = ds_context->comm_info[thread_id].get();
    scheduler->TryComm(thread_id);
    auto recv_offset = ExchangeSendSizes(send_offset, comm_info, rank, world_size, cuda_launch_lock);
    //while (*cuda_launch_lock > 0);
    CUDACHECK(cudaStreamSynchronize(stream));
    CHECK_EQ(*cuda_launch_lock, 0);
    scheduler->FinishComm();

    CUDACHECK(cudaStreamSynchronize(stream));
    auto host_recv_offset = recv_offset.CopyTo({kDLCPU, 0}, stream);
    IdType total_recv_size = host_recv_offset.Ptr<IdType>()[world_size] * expand_size;
    auto recvbuff = IdArray::Empty({total_recv_size}, input->dtype, dgl_context);

    // Exclusive all to all
    if(world_size > 1) {
      scheduler->TryComm(thread_id);
      CustomAlltoall(input.Ptr<void>(), send_offset.Ptr<IdType>(), recvbuff.Ptr<void>(), recv_offset.Ptr<IdType>(), type_bytes * expand_size, input->dtype.bits / 8, comm_info, rank, world_size, cuda_launch_lock);
      //while (*cuda_launch_lock > 0);
      CUDACHECK(cudaStreamSynchronize(stream));
      CHECK_EQ(*cuda_launch_lock, 0);
      scheduler->FinishComm();
    }

    // send data to myself in parallel
    auto* host_send_offset_ptr = host_send_offset.Ptr<IdType>();
    auto* host_recv_offset_ptr = host_recv_offset.Ptr<IdType>();

    int n_send_to_myself = host_send_offset_ptr[rank+1] - host_send_offset_ptr[rank];
    CUDACHECK(cudaMemcpyAsync(recvbuff.Ptr<void>() + host_recv_offset_ptr[rank] * expand_size * type_bytes, input.Ptr<void>() + host_send_offset_ptr[rank] * expand_size * type_bytes, n_send_to_myself * type_bytes * expand_size, cudaMemcpyDeviceToDevice, data_copy_stream));

    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamSynchronize(data_copy_stream));
    return {recvbuff, recv_offset};
  } else {
    // NCCL
    CHECK(send_offset->dtype.bits == 64);
    auto stream = CUDAThreadEntry::ThreadLocal()->stream;
    auto data_copy_stream = CUDAThreadEntry::ThreadLocal()->data_copy_stream;
    auto dgl_context = input->ctx;
    auto *ds_context = DSContext::Global();
    auto host_dgl_context = DLContext{kDLCPU, 0};
    auto send_sizes = Diff(send_offset);
    int comm_token = CUDAThreadEntry::ThreadLocal()->thread_id;
    IdArray recv_sizes = IdArray::Empty({world_size}, send_offset->dtype, dgl_context);
    IdArray range_seq = Range(0, world_size + 1, 64, host_dgl_context);
    int thread_id = CUDAThreadEntry::ThreadLocal()->thread_id;
    ncclComm_t nccl_comm = ds_context->nccl_comm[thread_id];

    scheduler->TryComm(thread_id);
    NCCLAllToAll<int64_t, ncclInt64>(send_sizes, range_seq, recv_sizes, range_seq, 1, rank, world_size, nccl_comm);
    CUDACHECK(cudaStreamSynchronize(stream));
    scheduler->FinishComm();

    auto host_send_offset = send_offset.CopyTo(host_dgl_context, stream);
    CUDACHECK(cudaStreamSynchronize(data_copy_stream));
    auto recv_offset = CumSum(recv_sizes, true);
    CUDACHECK(cudaStreamSynchronize(stream));
    IdArray host_recv_offset = recv_offset.CopyTo(host_dgl_context, stream);
    auto* host_recv_offset_ptr = host_recv_offset.Ptr<IdType>();
    int n_recv = host_recv_offset_ptr[world_size] * expand_size;
    auto recvbuff = IdArray::Empty({n_recv}, input->dtype, dgl_context);

    scheduler->TryComm(thread_id);
    if(input->dtype.bits == 32) {
      NCCLAllToAll<int, ncclInt32>(input, host_send_offset, recvbuff, host_recv_offset, expand_size, rank, world_size, nccl_comm);
    } else {
      NCCLAllToAll<int64_t, ncclInt64>(input, host_send_offset, recvbuff, host_recv_offset, expand_size, rank, world_size, nccl_comm);
    }
    CUDACHECK(cudaStreamSynchronize(stream));
    scheduler->FinishComm();
    CUDACHECK(cudaStreamSynchronize(data_copy_stream));
    return {recvbuff, recv_offset};
  }
}

}
}
