#include "ds_kernel.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>
#include <iostream>
#include "dmlc/logging.h"

#include <dgl/array.h>
#include <dgl/aten/csr.h>

#include "../memory_manager.h"
#include "./alltoall.h"
#include "../context.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace ds {

static constexpr int MAX_RECV_BUFFER_SIZE = 8 * 1000 * 250 * 10;

__global__
void _GidToLidKernel(IdType* global_ids, size_t size, IdType* min_vids, int rank) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int base = min_vids[rank];
  while (idx < size) {
    global_ids[idx] -= base;
    idx += BLOCK_NUM * BLOCK_SIZE;
  }
}

void ConvertGidToLid(IdArray global_ids, IdArray min_vids, int rank) {
  auto* global_ids_ptr = global_ids.Ptr<IdType>();
  auto* min_vids_ptr = min_vids.Ptr<IdType>();
  _GidToLidKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(global_ids_ptr, global_ids->shape[0], min_vids_ptr, rank);
}

__global__
void _LidToGidKernel(IdType* local_ids, size_t size, IdType* global_nid_map) {
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (idx < size) {
    local_ids[idx] = global_nid_map[local_ids[idx]];
    idx += BLOCK_NUM * BLOCK_SIZE;
  }
}

void ConvertLidToGid(IdArray local_ids, IdArray global_nid_map) {
  auto* local_ids_ptr = local_ids.Ptr<IdType>();
  auto* global_nid_map_ptr = global_nid_map.Ptr<IdType>();
  _LidToGidKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(local_ids_ptr, local_ids->shape[0], global_nid_map_ptr);
}

__global__
void _CountDeviceVerticesKernel(int device_cnt, 
                                uint64 *device_vid_base,
                                uint64 num_seed, 
                                uint64 *seeds,
                                uint64 *device_col_cnt) {
  __shared__ uint64 local_count[9];
  __shared__ uint64 device_vid[9];
  uint64 idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if (threadIdx.x <= device_cnt) {
    device_vid[threadIdx.x] = device_vid_base[threadIdx.x];
    local_count[threadIdx.x] = 0;
  }

  __syncthreads();

  uint64_t device_id, vid;
  while (idx < num_seed) {
    device_id = 0;
    vid = seeds[idx];
    while (device_id + 1 < device_cnt && device_vid[device_id + 1] <= vid) {
      ++device_id;
    }
    atomicAdd(&local_count[device_id], 1);
    idx += stride;
  }

  __syncthreads();

  if (threadIdx.x < device_cnt) {
    atomicAdd(&device_col_cnt[threadIdx.x], local_count[threadIdx.x]); 
  }

}

void Cluster(IdArray seeds, IdArray min_vids, int world_size, IdArray* send_sizes, IdArray* send_offset) {
  int n_seeds = seeds->shape[0];
  // thrust::device_ptr<IdType> seeds_ptr(seeds.Ptr<IdType>());
  // thrust::sort(seeds_ptr, seeds_ptr + n_seeds);
  auto dgl_ctx = seeds->ctx;
  *send_sizes = Full<int64_t>(0, world_size, dgl_ctx);
  *send_offset = Full<int64_t>(0, world_size + 1, dgl_ctx);
  // *send_sizes = MemoryManager::Global()->Full<int64_t>("SEND_SIZES", 0, world_size, dgl_ctx);
  // *send_offset = MemoryManager::Global()->Full<int64_t>("SEND_OFFSET", 0, world_size + 1, dgl_ctx);

  int n_threads = 1024;
  int n_blocks = (n_seeds + n_threads - 1) / n_threads;
  _CountDeviceVerticesKernel<<<n_blocks, n_threads>>>(world_size, min_vids.Ptr<IdType>(),
                                                        n_seeds, seeds.Ptr<IdType>(),
                                                        send_sizes->Ptr<IdType>());
  *send_offset = CumSum(*send_sizes, true);
  // thrust::exclusive_scan(thrust::device_ptr<IdType>(send_sizes->Ptr<IdType>()), 
  //                        thrust::device_ptr<IdType>(send_sizes->Ptr<IdType>()) + world_size + 1, send_offset->Ptr<IdType>());  
}

void AllToAll(IdArray send_buffer, IdArray send_offset, IdArray recv_buffer, IdArray recv_offset, int expand_size, int rank, int world_size, ncclComm_t nccl_comm) {
  IdType* send_buffer_ptr = send_buffer.Ptr<IdType>();
  IdType* recv_buffer_ptr = recv_buffer.Ptr<IdType>();
  IdType* send_offset_ptr = send_offset.Ptr<IdType>();
  IdType* recv_offset_ptr = recv_offset.Ptr<IdType>();
  cudaMemcpy(recv_buffer_ptr + recv_offset_ptr[rank] * expand_size, send_buffer_ptr + send_offset_ptr[rank] * expand_size, (send_offset_ptr[rank + 1] - send_offset_ptr[rank]) * expand_size * sizeof(IdType), cudaMemcpyDeviceToDevice);
  ncclGroupStart();
  for(int r = 0; r < world_size; ++r) {
    if(r != rank) {
      IdType send_size = (send_offset_ptr[r+1] - send_offset_ptr[r]) * expand_size;
      IdType send_ptr = send_offset_ptr[r] * expand_size;
      IdType recv_size = (recv_offset_ptr[r+1] - recv_offset_ptr[r]) * expand_size;
      IdType recv_ptr = recv_offset_ptr[r] * expand_size;
      ncclSend(send_buffer_ptr + send_ptr, send_size, ncclUint64, r, nccl_comm, 0);
      ncclRecv(recv_buffer_ptr + recv_ptr, recv_size, ncclUint64, r, nccl_comm, 0);
    }
  }
  ncclGroupEnd();
}

void AllToAllV2(IdArray send_buffer, IdArray send_offset, IdArray* recv_buffer, IdArray* host_recv_offset, int rank, int world_size, const std::string& scope) {
  auto* ds_context = DSContext::Global();
  auto dgl_context = send_buffer->ctx;
  *recv_buffer = IdArray::Empty({MAX_RECV_BUFFER_SIZE}, send_buffer->dtype, dgl_context);
  IdArray recv_offset = IdArray::Empty({world_size + 1}, send_buffer->dtype, dgl_context);
  // *recv_buffer = MemoryManager::Global()->Empty(scope + "_RECV_BUFFER", {MAX_RECV_BUFFER_SIZE}, send_buffer->dtype, dgl_context);
  // IdArray recv_offset = MemoryManager::Global()->Empty(scope + "_RECV_OFFSET", {world_size + 1}, send_buffer->dtype, dgl_context);

  Alltoall(send_buffer.Ptr<IdType>(), send_offset.Ptr<IdType>(), recv_buffer->Ptr<IdType>(), recv_offset.Ptr<IdType>(), &ds_context->comm_info, rank, world_size);

  *host_recv_offset = recv_offset.CopyTo({kDLCPU, 0});
  IdType* host_recv_offset_ptr = host_recv_offset->Ptr<IdType>();
  CHECK_LE(host_recv_offset_ptr[world_size], MAX_RECV_BUFFER_SIZE);
  *recv_buffer = recv_buffer->CreateView({(signed long) host_recv_offset_ptr[world_size]}, send_buffer->dtype);
}

void Shuffle(IdArray seeds, IdArray host_send_offset, IdArray send_sizes, int rank, int world_size, ncclComm_t nccl_comm, IdArray* frontier, IdArray* host_recv_offset) {
  auto dgl_context = seeds->ctx;
  auto host_dgl_context = DLContext{kDLCPU, 0};
  IdArray recv_sizes = IdArray::Empty({world_size}, seeds->dtype, dgl_context);
  // IdArray recv_sizes = MemoryManager::Global()->Empty("RECV_SIZES", {world_size}, seeds->dtype, dgl_context);
  IdArray range_seq = Range(0, world_size + 1, 64, host_dgl_context);
  AllToAll(send_sizes, range_seq, recv_sizes, range_seq, 1, rank, world_size, nccl_comm);

  IdArray host_recv_sizes = recv_sizes.CopyTo(host_dgl_context);
  *host_recv_offset = Full<int64_t>(0, world_size + 1, host_dgl_context);
  // *host_recv_offset = MemoryManager::Global()->Full<int64_t>("HOST_RECV_OFFSET", 0, world_size + 1, host_dgl_context);
  auto* host_recv_offset_ptr = host_recv_offset->Ptr<IdType>();
  auto* host_recv_sizes_ptr = host_recv_sizes.Ptr<IdType>();
  host_recv_offset_ptr[0] = 0;
  for(int i = 1; i <= world_size; ++i) {
    host_recv_offset_ptr[i] = host_recv_offset_ptr[i-1] + host_recv_sizes_ptr[i-1];
  }
  int n_frontier = host_recv_offset_ptr[world_size];
  *frontier = IdArray::Empty({n_frontier}, seeds->dtype, dgl_context);
  // *frontier = MemoryManager::Global()->Empty("FRONTIER", {n_frontier}, seeds->dtype, dgl_context);
  AllToAll(seeds, host_send_offset, *frontier, *host_recv_offset, 1, rank, world_size, nccl_comm);
}

void ShuffleV2(IdArray seeds, IdArray send_offset, int rank, int world_size, IdArray* frontier, IdArray* host_recv_offset) {
  AllToAllV2(seeds, send_offset, frontier, host_recv_offset, rank, world_size, "SHUFFLEV2");
}


template <int BLOCK_ROWS>
__global__ void _CSRRowWiseSampleReplaceKernel(
    int fanout, 
    uint64 num_frontier, 
    uint64 *frontier,
    uint64 *in_ptr, 
    uint64 *in_index,
    uint64 *edge_index,
    uint64 *out_ptr, 
    uint64 *out_index,
    uint64 *out_edges) {
  assert(blockDim.x == WARP_SIZE);
  constexpr int NUM_RNG = ((WARP_SIZE*BLOCK_ROWS)+255)/256;
  // __shared__ curandState rng_array[NUM_RNG];
  __shared__ curandState rng_array[WARP_SIZE*BLOCK_ROWS];
  assert(blockDim.x >= NUM_RNG);
  // int rand_seed = 0;
  int pos = threadIdx.x + WARP_SIZE * threadIdx.y;
  int rand_seed = pos;
  // if (threadIdx.y == 0 && threadIdx.x < NUM_RNG) {
  //   curand_init(rand_seed, 0, threadIdx.x, rng_array+threadIdx.x);
  // }
  // curand_init(rand_seed, 0, threadIdx.x, rng_array+threadIdx.x);
  curand_init(rand_seed, 0, threadIdx.x, rng_array+pos);
  __syncthreads();
  // curandState * const rng = rng_array+((threadIdx.x+WARP_SIZE*threadIdx.y)/256);
  curandState * const rng = rng_array + pos;

  uint64 out_row = blockIdx.x*blockDim.y+threadIdx.y;
  while (out_row < num_frontier) {
    const uint64 row = frontier[out_row];
    const uint64 in_row_start = in_ptr[row];
    const uint64 out_row_start = out_ptr[out_row];
    const uint64 deg = in_ptr[row + 1] - in_row_start;
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const uint64 edge = curand(rng) % deg;
      const uint64 out_idx = out_row_start + idx;
      out_index[out_idx] = in_index[in_row_start + edge];
      //out_edges[out_idx] = edge_index[in_row_start + edge];
    }
    out_row += gridDim.x * blockDim.y;
  }
}

void SampleNeighbors(IdArray frontier, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges) {
  auto dgl_ctx = frontier->ctx;
  int n_frontier = frontier->shape[0];
  IdArray edge_offset = Full<int64_t>(fanout, n_frontier + 1, dgl_ctx);
  // IdArray edge_offset = MemoryManager::Global()->Full<int64_t>("EDGE_OFFSET", fanout, n_frontier + 1, dgl_ctx);
  auto edge_offset_ptr = thrust::device_ptr<IdType>(edge_offset.Ptr<IdType>());
  thrust::exclusive_scan(edge_offset_ptr, edge_offset_ptr + n_frontier + 1, edge_offset_ptr);
  *neighbors = IdArray::Empty({n_frontier * fanout}, frontier->dtype, dgl_ctx);
  *edges = IdArray::Empty({n_frontier * fanout}, frontier->dtype, dgl_ctx);
  // *neighbors = MemoryManager::Global()->Empty("NEIGHBORS", {n_frontier * fanout}, frontier->dtype, dgl_ctx);
  // *edges = MemoryManager::Global()->Empty("EDGES", {n_frontier * fanout}, frontier->dtype, dgl_ctx);

  constexpr int BLOCK_ROWS = 128 / WARP_SIZE;
  const dim3 block(WARP_SIZE, BLOCK_ROWS);
  const dim3 grid((n_frontier + block.y - 1) / block.y);
  _CSRRowWiseSampleReplaceKernel<BLOCK_ROWS><<<grid, block>>>(
    fanout, n_frontier, frontier.Ptr<IdType>(), csr_mat.indptr.Ptr<IdType>(), csr_mat.indices.Ptr<IdType>(), csr_mat.data.Ptr<IdType>(),
    edge_offset.Ptr<IdType>(), neighbors->Ptr<IdType>(), edges->Ptr<IdType>()
  );
  // CUDACHECK(cudaDeviceSynchronize());
}

void SampleNeighborsV2(IdArray frontier, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges) {
  auto coo = CSRRowWiseSampling(csr_mat, frontier, fanout, NullArray(frontier->dtype, frontier->ctx));
  *neighbors = coo.col;
}

void Reshuffle(IdArray neighbors, int fanout, int n_seeds, IdArray host_shuffle_send_offset, IdArray host_shuffle_recv_offset, int rank, int world_size, ncclComm_t nccl_comm, IdArray* reshuffled_neighbors) {
  int shuffle_send_size = host_shuffle_send_offset.Ptr<IdType>()[world_size];
  *reshuffled_neighbors = IdArray::Empty({shuffle_send_size * fanout}, neighbors->dtype, neighbors->ctx);
  // *reshuffled_neighbors = MemoryManager::Global()->Empty("RESHUFFLED_NEIGHBORS", {shuffle_send_size * fanout}, neighbors->dtype, neighbors->ctx);
  AllToAll(neighbors, host_shuffle_recv_offset, *reshuffled_neighbors, host_shuffle_send_offset, fanout, rank, world_size, nccl_comm);
}

void ReshuffleV2(IdArray neighbors, int fanout, IdArray host_shuffle_recv_offset, int rank, int world_size, IdArray* reshuffled_neighbors) {
  auto dgl_context = neighbors->ctx;
  auto* host_shuffle_recv_offset_ptr = host_shuffle_recv_offset.Ptr<IdType>();
  for(int i = 0; i <= world_size; ++i) {
    host_shuffle_recv_offset_ptr[i] *= fanout;
  }
  auto shuffle_recv_offset = host_shuffle_recv_offset.CopyTo(dgl_context);
  IdArray recv_offset;
  AllToAllV2(neighbors, shuffle_recv_offset, reshuffled_neighbors, &recv_offset, rank, world_size, "RESHUFFLEV2");
}

template <int BLOCK_ROWS>
__global__ void _CSRRowWiseReplicateKernel(
    uint64 size,
    uint64 *src,
    uint64 *dst,
    int fanout) {
  uint64 out_row = blockIdx.x * blockDim.y + threadIdx.y;
  while (out_row < size) {
    const uint64 out_row_start = out_row * fanout;
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      dst[out_row_start + idx] = src[out_row];
    }
    out_row += gridDim.x * blockDim.y;
  }
}

void Replicate(IdArray src, IdArray *dst, int fanout) {
  auto *src_ptr = src.Ptr<IdType>();
  auto dgl_ctx = src->ctx;
  int n_seeds = src->shape[0];
  *dst = IdArray::Empty({n_seeds * fanout}, src->dtype, dgl_ctx);
  // *dst = MemoryManager::Global()->Empty("DESTINATION", {n_seeds * fanout}, src->dtype, dgl_ctx);
  auto *dst_ptr = (*dst).Ptr<IdType>();

  constexpr int BLOCK_ROWS = 128 / WARP_SIZE;
  const dim3 block(WARP_SIZE, BLOCK_ROWS);
  const dim3 grid((n_seeds + block.y - 1) / block.y);
  _CSRRowWiseReplicateKernel<BLOCK_ROWS><<<grid, block>>>(
    n_seeds, src_ptr, dst_ptr, fanout
  );
  CUDACHECK(cudaDeviceSynchronize());
}

void SampleNeighborsUVA(IdArray frontier, IdArray row_idx, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges) {
  auto dgl_ctx = frontier->ctx;
  int n_frontier = frontier->shape[0];
  IdArray edge_offset = MemoryManager::Global()->Full<int64_t>("EDGE_OFFSET", fanout, n_frontier + 1, dgl_ctx);
  auto edge_offset_ptr = thrust::device_ptr<IdType>(edge_offset.Ptr<IdType>());
  thrust::exclusive_scan(edge_offset_ptr, edge_offset_ptr + n_frontier + 1, edge_offset_ptr);
  *neighbors = MemoryManager::Global()->Empty("NEIGHBORS", {n_frontier * fanout}, frontier->dtype, dgl_ctx);
  *edges = MemoryManager::Global()->Empty("EDGES", {n_frontier * fanout}, frontier->dtype, dgl_ctx);

  constexpr int BLOCK_ROWS = 128 / WARP_SIZE;
  const dim3 block(WARP_SIZE, BLOCK_ROWS);
  const dim3 grid((n_frontier + block.y - 1) / block.y);
  _CSRRowWiseSampleReplaceKernel<BLOCK_ROWS><<<grid, block>>>(
    fanout, n_frontier, frontier.Ptr<IdType>(), row_idx.Ptr<IdType>(), csr_mat.indices.Ptr<IdType>(), csr_mat.data.Ptr<IdType>(),
    edge_offset.Ptr<IdType>(), neighbors->Ptr<IdType>(), edges->Ptr<IdType>()
  );
  CUDACHECK(cudaDeviceSynchronize());
}

}
}
