#include "ds_kernel.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>
#include <cmath>
#include <iostream>
#include "dmlc/logging.h"
#include <thread>

#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include <dgl/runtime/device_api.h>

#include "../memory_manager.h"
#include "./alltoall.h"
#include "../context.h"
#include "./scan.h"

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
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _GidToLidKernel<<<BLOCK_NUM, BLOCK_SIZE, 0, thr_entry->stream>>>(global_ids_ptr, global_ids->shape[0], min_vids_ptr, rank);
  CUDACHECK(cudaGetLastError());
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
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _LidToGidKernel<<<BLOCK_NUM, BLOCK_SIZE, 0, thr_entry->stream>>>(local_ids_ptr, local_ids->shape[0], global_nid_map_ptr);
}

__global__
void _CountDeviceVerticesKernel(int device_cnt, 
                                const IdType *device_vid_base,
                                IdType num_seed, 
                                const IdType *seeds,
                                IdType *device_col_cnt,
                                IdType *part_ids) {
  __shared__ IdType local_count[9];
  __shared__ IdType device_vid[9];
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if (threadIdx.x <= device_cnt) {
    device_vid[threadIdx.x] = device_vid_base[threadIdx.x];
    local_count[threadIdx.x] = 0;
  }

  __syncthreads();

  IdType device_id, vid;
  while (idx < num_seed) {
    device_id = 0;
    vid = seeds[idx];
    while (device_id + 1 < device_cnt && device_vid[device_id + 1] <= vid) {
      ++device_id;
    }
    part_ids[idx] = device_id;
    atomicAdd((unsigned long long *)(local_count + device_id), 1);
    idx += stride;
  }

  __syncthreads();

  if (threadIdx.x < device_cnt) {
    atomicAdd((unsigned long long*)(device_col_cnt + threadIdx.x), local_count[threadIdx.x]); 
  }

}

std::tuple<IdArray, IdArray, IdArray, IdArray> Partition(IdArray seeds, IdArray min_vids, int world_size) {
  auto dgl_ctx = seeds->ctx;
  IdArray part_sizes = Full<int64_t>(0, world_size, dgl_ctx);
  IdArray part_ids = IdArray::Empty({seeds->shape[0]}, seeds->dtype, seeds->ctx);
  int n_threads = 1024;
  int n_blocks = (seeds->shape[0] + n_threads - 1) / n_threads;
  if (n_blocks == 0) n_blocks = 1;
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _CountDeviceVerticesKernel<<<n_blocks, n_threads, 0, thr_entry->stream>>>(world_size, min_vids.Ptr<IdType>(),
                                                                            seeds->shape[0], seeds.Ptr<IdType>(),
                                                                            part_sizes.Ptr<IdType>(), part_ids.Ptr<IdType>());
  IdArray part_offset = CumSum(part_sizes, true);
  IdArray sorted, index;
  std::tie(sorted, index) = MultiWayScan(seeds, part_offset, part_ids, world_size);
  return {sorted, index, part_sizes, part_offset};
}

IdArray Partition(IdArray seeds, IdArray min_vids) {
  auto dgl_ctx = seeds->ctx;
  int world_size = min_vids->shape[0] - 1;
  IdArray part_sizes = Full<int64_t>(0, world_size, dgl_ctx);
  IdArray part_ids = IdArray::Empty({seeds->shape[0]}, seeds->dtype, seeds->ctx);
  int n_threads = 1024;
  int n_blocks = (seeds->shape[0] + n_threads - 1) / n_threads;
  if (n_blocks == 0) n_blocks = 1;
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _CountDeviceVerticesKernel<<<n_blocks, n_threads, 0, thr_entry->stream>>>(world_size, min_vids.Ptr<IdType>(),
                                                                            seeds->shape[0], seeds.Ptr<IdType>(),
                                                                            part_sizes.Ptr<IdType>(), part_ids.Ptr<IdType>());
  IdArray part_offset = CumSum(part_sizes, true);
  IdArray sorted, index;
  std::tie(sorted, index) = MultiWayScan(seeds, part_offset, part_ids, world_size);
  return sorted;
}

void Cluster(int rank, IdArray seeds, IdArray min_vids, int world_size, IdArray* send_sizes, IdArray* send_offset) {
  auto dgl_ctx = seeds->ctx;
  *send_sizes = Full<int64_t>(0, world_size, dgl_ctx);
  IdArray part_ids = IdArray::Empty({seeds->shape[0]}, seeds->dtype, seeds->ctx);
  int n_threads = 1024;
  int n_blocks = (seeds->shape[0] + n_threads - 1) / n_threads;
  if (n_blocks == 0) n_blocks = 1;
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _CountDeviceVerticesKernel<<<n_blocks, n_threads, 0, thr_entry->stream>>>(world_size, min_vids.Ptr<IdType>(),
                                                                            seeds->shape[0], seeds.Ptr<IdType>(),
                                                                            send_sizes->Ptr<IdType>(), part_ids.Ptr<IdType>());
  *send_offset = CumSum(*send_sizes, true);
}

template <int BLOCK_ROWS>
__global__ void _CSRRowWiseSampleReplaceKernel(
    int fanout, 
    IdType num_frontier, 
    IdType *frontier,
    IdType *in_ptr, 
    IdType *in_index,
    IdType *out_ptr, 
    IdType *out_index) {
  assert(blockDim.x == WARP_SIZE);
  constexpr int NUM_RNG = ((WARP_SIZE*BLOCK_ROWS)+255)/256;
  __shared__ curandState rng_array[WARP_SIZE*BLOCK_ROWS];
  assert(blockDim.x >= NUM_RNG);
  int pos = threadIdx.x + WARP_SIZE * threadIdx.y;
  int rand_seed = pos;
  curand_init(rand_seed, 0, threadIdx.x, rng_array+pos);
  __syncthreads();
  curandState * const rng = rng_array + pos;

  IdType out_row = blockIdx.x*blockDim.y+threadIdx.y;
  while (out_row < num_frontier) {
    const IdType row = frontier[out_row];
    const IdType in_row_start = in_ptr[row];
    const IdType out_row_start = out_ptr[out_row];
    const IdType deg = in_ptr[row + 1] - in_row_start;
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const IdType edge = curand(rng) % deg;
      const IdType out_idx = out_row_start + idx;
      out_index[out_idx] = in_index[in_row_start + edge];
    }
    out_row += gridDim.x * blockDim.y;
  }
}

void SampleNeighbors(IdArray frontier, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges) {
  auto dgl_ctx = frontier->ctx;
  int n_frontier = frontier->shape[0];
  IdArray edge_offset = Full<int64_t>(fanout, n_frontier, dgl_ctx);
  // IdArray edge_offset = MemoryManager::Global()->Full<int64_t>("EDGE_OFFSET", fanout, n_frontier + 1, dgl_ctx);
  edge_offset = CumSum(edge_offset, true);
  // auto edge_offset_ptr = thrust::device_ptr<IdType>(edge_offset.Ptr<IdType>());
  // thrust::exclusive_scan(edge_offset_ptr, edge_offset_ptr + n_frontier + 1, edge_offset_ptr);
  // CUDACHECK(cudaDeviceSynchronize());
  *neighbors = IdArray::Empty({n_frontier * fanout}, frontier->dtype, dgl_ctx);
  *edges = IdArray::Empty({n_frontier * fanout}, frontier->dtype, dgl_ctx);
  // *neighbors = MemoryManager::Global()->Empty("NEIGHBORS", {n_frontier * fanout}, frontier->dtype, dgl_ctx);
  // *edges = MemoryManager::Global()->Empty("EDGES", {n_frontier * fanout}, frontier->dtype, dgl_ctx);

  constexpr int BLOCK_ROWS = 128 / WARP_SIZE;
  const dim3 block(WARP_SIZE, BLOCK_ROWS);
  const dim3 grid((n_frontier + block.y - 1) / block.y);
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  if(grid.x > 0) {
    _CSRRowWiseSampleReplaceKernel<BLOCK_ROWS><<<grid, block, 0, thr_entry->stream>>>(
      fanout, n_frontier, frontier.Ptr<IdType>(), csr_mat.indptr.Ptr<IdType>(), csr_mat.indices.Ptr<IdType>(),
      edge_offset.Ptr<IdType>(), neighbors->Ptr<IdType>()
    );
  }
  // CUDACHECK(cudaDeviceSynchronize());
}

void SampleNeighborsV2(IdArray frontier, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges) {
  if(frontier->shape[0] == 0) {
    *neighbors = NullArray(frontier->dtype, frontier->ctx);
  } else {
    auto coo = CSRRowWiseSampling(csr_mat, frontier, fanout, NullArray(frontier->dtype, frontier->ctx));
    *neighbors = coo.col;
  }
}

template <class T>
__global__
void _RemapKernel(T* dst, T* src, IdType* index, int size, int fanout) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  while(tid < size * fanout) {
    IdType vid = tid / fanout;
    IdType dst_pos = index[vid] * fanout + tid % fanout;
    dst[dst_pos] = src[tid];
    tid += stride;
  }
}

IdArray Remap(IdArray neighbors, IdArray index, int fanout) {
  CHECK_EQ(neighbors->shape[0], index->shape[0] * fanout);
  IdArray ret = IdArray::Empty({neighbors->shape[0]}, neighbors->dtype, neighbors->ctx);
  int n_threads = 512;
  int n_blocks = 128;
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  if (neighbors->dtype.bits == 64) {
    _RemapKernel<<<n_blocks, n_threads, 0, thr_entry->stream>>>(ret.Ptr<IdType>(), neighbors.Ptr<IdType>(), index.Ptr<IdType>(), index->shape[0], fanout);
  } else {
    _RemapKernel<<<n_blocks, n_threads, 0, thr_entry->stream>>>(ret.Ptr<DataType>(), neighbors.Ptr<DataType>(), index.Ptr<IdType>(), index->shape[0], fanout);
  }
  return ret;
}

template <int BLOCK_ROWS>
__global__ void _CSRRowWiseReplicateKernel(
    IdType size,
    IdType *src,
    IdType *dst,
    int fanout) {
  IdType out_row = blockIdx.x * blockDim.y + threadIdx.y;
  while (out_row < size) {
    const IdType out_row_start = out_row * fanout;
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
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _CSRRowWiseReplicateKernel<BLOCK_ROWS><<<grid, block, 0, thr_entry->stream>>>(
    n_seeds, src_ptr, dst_ptr, fanout
  );
}

void SampleNeighborsUVA(IdArray frontier, IdArray row_idx, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges) {
  auto dgl_ctx = frontier->ctx;
  int n_frontier = frontier->shape[0];
  // IdArray edge_offset = IdArray::Empty({n_frontier + 1}, frontier->dtype, dgl_ctx);
  IdArray edge_offset = Full<IdType>(fanout, n_frontier, dgl_ctx);
  edge_offset = CumSum(edge_offset, true);
  *neighbors = IdArray::Empty({n_frontier * fanout}, frontier->dtype, dgl_ctx);

  constexpr int BLOCK_ROWS = 512 / WARP_SIZE;
  const dim3 block(WARP_SIZE, BLOCK_ROWS);
  const dim3 grid((n_frontier + 512 - 1) / 512);
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  if(grid.x > 0) {
    _CSRRowWiseSampleReplaceKernel<BLOCK_ROWS><<<grid, block, 0, thr_entry->stream>>>(
      fanout, n_frontier, frontier.Ptr<IdType>(), row_idx.Ptr<IdType>(), csr_mat.indices.Ptr<IdType>(),
      edge_offset.Ptr<IdType>(), neighbors->Ptr<IdType>()
    );
  }
}

template <int BLOCK_ROWS>
__global__ void _CSRRowWiseLoadSubtensorKernel(
    IdType dim, 
    IdType num_frontier, 
    IdType *frontier,
    DataType *features,
    DataType *features_to_send) {
  assert(blockDim.x == WARP_SIZE);
  IdType out_row = blockIdx.x*blockDim.y+threadIdx.y;
  while (out_row < num_frontier) {
    const IdType row = frontier[out_row];
    const IdType in_row_start = row * dim;
    const IdType out_row_start = out_row * dim;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
      features_to_send[out_row_start + idx] = features[in_row_start + idx];
    }
    out_row += gridDim.x * blockDim.y;
  }
}

template <int BLOCK_ROWS>
__global__ void _CSRRowWiseLoadSubtensorAlignedKernel(
    IdType dim, 
    IdType num_frontier, 
    IdType *frontier,
    DataType *features,
    DataType *features_to_send) {
  IdType out_row = blockIdx.x*blockDim.y+threadIdx.y;
  while (out_row < num_frontier) {
    const IdType row = frontier[out_row];
    const IdType origin_in_row_start = row * dim;
    const IdType out_row_start = out_row * dim;
    const IdType in_row_start = origin_in_row_start & ~0x3;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
      if (in_row_start + idx >= origin_in_row_start) {
        features_to_send[out_row_start + idx] = features[in_row_start + idx];
      }
    }
    out_row += gridDim.x * blockDim.y;
  }
}

void LoadFeature(IdArray frontier, IdArray features, IdArray *features_to_send) {
  auto dgl_ctx = features->ctx;
  int n_frontier = frontier->shape[0], dim = features->shape[1];
  *features_to_send = IdArray::Empty({n_frontier * dim}, features->dtype, dgl_ctx);
  constexpr int BLOCK_X = 64;
  constexpr int BLOCK_ROWS = 16;
  const dim3 block(BLOCK_X, BLOCK_ROWS);
  int BLOCK_NUM = n_frontier / BLOCK_ROWS;
  const dim3 grid(BLOCK_NUM);
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  if (grid.x > 0) {
    _CSRRowWiseLoadSubtensorAlignedKernel<BLOCK_ROWS><<<grid, block, 0, thr_entry->stream>>>(
      dim, n_frontier, frontier.Ptr<IdType>(), features.Ptr<DataType>(), features_to_send->Ptr<DataType>()
    );
    CUDACHECK(cudaGetLastError());
  }
}

}
}
