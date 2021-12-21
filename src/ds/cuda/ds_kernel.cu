#include "ds_kernel.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>
#include <iostream>

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

__global__
void _MinusKernel(uint64 num_id, uint64 *global_id, uint64 *d_device_vids, int rank) {
  uint64 idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (idx < num_id) {
    global_id[idx] -= d_device_vids[rank];
    idx += BLOCK_NUM * BLOCK_SIZE;
  }
}

void ConvertGidToLid(uint64 num_id, uint64 *global_id, uint64 *d_device_vids, int rank) {
  _MinusKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(num_id, global_id, d_device_vids, rank);
}

__global__
void _AddKernel(uint64 num_id, uint64 *local_id, uint64 *d_device_vids, int rank) {
  uint64 idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (idx < num_id) {
    local_id[idx] += d_device_vids[rank];
    idx += BLOCK_NUM * BLOCK_SIZE;
  }
}

void ConvertLidToGid(uint64 num_id, uint64 *local_id, uint64 *d_device_vids, int rank) {
  _AddKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(num_id, local_id, d_device_vids, rank);
}

__global__
void _CountDeviceVerticesKernel(int device_cnt, 
                                uint64 *device_vid_base,
                                uint64 num_seed, 
                                uint64 *seeds, 
                                uint64 num_picks, 
                                uint64 *device_col_cnt) {
  __shared__ uint64 local_count[9];
  __shared__ uint64 device_vid[9];
  assert(blockDim.x == BLOCK_SIZE);
  uint64 idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (threadIdx.x <= device_cnt) {
    device_vid[threadIdx.x] = device_vid_base[threadIdx.x];
    local_count[threadIdx.x] = 0;
  }

  __syncthreads();

  uint64_t device_id, vid;
  while (idx < num_seed) {
    device_id = 0;
    vid = seeds[idx];
    while (device_vid[device_id] <= vid) {
      ++device_id;
    }
    --device_id;
    atomicAdd(&local_count[device_id], num_picks);
    idx += BLOCK_NUM * BLOCK_SIZE;
  }

  __syncthreads();

  if (threadIdx.x < device_cnt) {
    atomicAdd(&device_col_cnt[threadIdx.x], local_count[threadIdx.x]); 
  }

}

void Cluster( int device_cnt, uint64 *device_vid_base,
              uint64 num_seed, uint64 *seeds, int fanout,
              uint64 *device_col_ptr, uint64 *device_col_cnt, 
              uint64 **d_device_col_cnt) {
  thrust::device_ptr<uint64> d_seeds(seeds);
  thrust::sort(d_seeds, d_seeds + num_seed);

  uint64 *d_device_col_ptr;
  CUDACHECK(cudaMalloc((void**)&(*d_device_col_cnt), sizeof(uint64) * device_cnt));
  CUDACHECK(cudaMemset((*d_device_col_cnt), 0, sizeof(uint64) * device_cnt));
  CUDACHECK(cudaMalloc((void**)&d_device_col_ptr, sizeof(uint64) * (device_cnt + 1)));
  CUDACHECK(cudaMemset(d_device_col_ptr, 0, sizeof(uint64) * (device_cnt + 1)));

  _CountDeviceVerticesKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(device_cnt, device_vid_base,
                                                        num_seed, seeds, fanout,
                                                        *d_device_col_cnt);
  
  thrust::exclusive_scan(thrust::device_ptr<uint64>(*d_device_col_cnt), 
                         thrust::device_ptr<uint64>(*d_device_col_cnt) + device_cnt + 1, d_device_col_ptr);  
  CUDACHECK(cudaMemcpy(device_col_ptr, d_device_col_ptr, sizeof(uint64)*(device_cnt + 1), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(device_col_cnt, *d_device_col_cnt, sizeof(uint64)*device_cnt, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(d_device_col_ptr));
}

template <typename T>
T* Convert(thrust::device_ptr<T> ptr) {
  return thrust::raw_pointer_cast(ptr);
}

void Scatter(thrust::device_ptr<uint64> send_buffer, uint64 *send_offset,
             thrust::device_ptr<uint64> recv_buffer, uint64 *recv_offset,
             int group_size, int rank, ncclComm_t &comm, cudaStream_t &s) {
  CUDACHECK(cudaMemcpy(Convert(recv_buffer + recv_offset[rank]), 
                       Convert(send_buffer + send_offset[rank]),
                       (send_offset[rank + 1] - send_offset[rank]) * sizeof(uint64), 
                       cudaMemcpyDeviceToDevice));
  for (int i = 0; i < group_size; i++) {
    if (i == rank) {
      for (int j = 0; j < group_size; j++) {
        if (j != rank) {
          NCCLCHECK(ncclSend((const void*)Convert(send_buffer + send_offset[j]),
                             send_offset[j + 1] - send_offset[j],
                             ncclUint64, j, comm, s));
        }
      }
    } else {
      NCCLCHECK(ncclRecv((void*)Convert(recv_buffer + recv_offset[i]),
                          recv_offset[i + 1] - recv_offset[i],
                          ncclUint64, i, comm, s));
    }
  }
  CUDACHECK(cudaStreamSynchronize(s));
}

void Shuffle(int device_cnt, uint64 *device_col_ptr, uint64 *device_col_cnt, uint64 *d_device_col_cnt,
             uint64 *seeds,
             uint64 &num_frontier, uint64 *device_offset, uint64 **frontier, 
             int rank, ncclComm_t &comm) {
  uint64 *d_device_recv_cnt;
  cudaMalloc((void**)&d_device_recv_cnt, sizeof(uint64)*device_cnt);
  cudaStream_t s = 0;
  thrust::device_ptr<uint64> d_d_device_recv_cnt(d_device_recv_cnt);
  thrust::device_ptr<uint64> d_d_device_col_cnt(d_device_col_cnt);
  uint64 seq[device_cnt + 1];
  for (int i=0; i<=device_cnt; i++) {
    seq[i] = i;
  }
  Scatter(d_d_device_col_cnt, seq,
          d_d_device_recv_cnt, seq,
          device_cnt, rank, comm, s);
  
  uint64 device_recv_cnt[device_cnt];
  CUDACHECK(cudaMemcpy(device_recv_cnt, d_device_recv_cnt, sizeof(uint64) * device_cnt, cudaMemcpyDeviceToHost));
  device_offset[0] = 0;
  for (int i = 1; i <= device_cnt; i++) {
    device_offset[i] = device_offset[i-1] + device_recv_cnt[i-1];
  }
  num_frontier = device_offset[device_cnt];
  CUDACHECK(cudaMalloc((void**)&(*frontier), sizeof(uint64)*num_frontier));
  thrust::device_ptr<uint64> d_seeds(seeds), d_frontier(*frontier);

  Scatter(d_seeds, device_col_ptr,
          d_frontier, device_offset,
          device_cnt, rank, comm, s);
}

template <int BLOCK_ROWS>
__global__ void _CSRRowWiseSampleReplaceKernel(
    int fanout, 
    uint64 num_frontier, 
    uint64 *frontier,
    uint64 *in_ptr, 
    uint64 *in_index,
    uint64 *out_ptr, 
    uint64 *out_index) {
  assert(blockDim.x == WARP_SIZE);
  constexpr int NUM_RNG = ((WARP_SIZE*BLOCK_ROWS)+255)/256;
  __shared__ curandState rng_array[NUM_RNG];
  assert(blockDim.x >= NUM_RNG);
  int rand_seed = 0;
  if (threadIdx.y == 0 && threadIdx.x < NUM_RNG) {
    curand_init(rand_seed, 0, threadIdx.x, rng_array+threadIdx.x);
  }
  __syncthreads();
  curandState * const rng = rng_array+((threadIdx.x+WARP_SIZE*threadIdx.y)/256);

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
    }
    out_row += gridDim.x * blockDim.y;
  }
}

void SampleNeighbors(int fanout, uint64 num_frontier, uint64 *frontier,
                     uint64 *in_ptr, uint64 *in_index,
                     uint64 **out_index) {
  uint64 *out_ptr;
  cudaMalloc((void**)&out_ptr, sizeof(uint64) * (num_frontier + 1));
  cudaMalloc((void**)&(*out_index), sizeof(uint64) * num_frontier * fanout);
  thrust::device_ptr<uint64> d_out_ptr(out_ptr);
  thrust::fill(d_out_ptr, d_out_ptr + num_frontier, uint64(fanout));
  thrust::exclusive_scan(d_out_ptr, d_out_ptr + num_frontier + 1, d_out_ptr);
  constexpr int BLOCK_ROWS = 128 / WARP_SIZE;
  const dim3 block(WARP_SIZE, BLOCK_ROWS);
  const dim3 grid((num_frontier + block.y - 1) / block.y);
  cudaStream_t s = 0;
  _CSRRowWiseSampleReplaceKernel<BLOCK_ROWS><<<grid, block, 0, s>>>(
    fanout, num_frontier, frontier, in_ptr, in_index, out_ptr, *out_index
  );
  cudaStreamSynchronize(s);
}

void Reshuffle(int fanout, int num_devices, uint64 *device_offset, uint64 *cols,
               uint64 *device_col_ptr, uint64 num_seed, uint64 **out_ptr, uint64 **out_cols,
               int rank, ncclComm_t &comm) {
  cudaMalloc((void**)&(*out_ptr), sizeof(uint64) * (num_seed + 1));
  thrust::device_ptr<uint64> d_out_ptr(*out_ptr);
  thrust::fill(d_out_ptr, d_out_ptr + num_seed, fanout);
  thrust::exclusive_scan(d_out_ptr, d_out_ptr + num_seed + 1, d_out_ptr);
  cudaMalloc((void**)&(*out_cols), sizeof(uint64) * num_seed * fanout);
  thrust::device_ptr<uint64> d_cols(cols), d_out_cols(*out_cols);
  cudaStream_t s = 0;
  Scatter(thrust::device_ptr<uint64>(cols), device_offset, 
          thrust::device_ptr<uint64>(*out_cols), device_col_ptr,
          num_devices, rank, comm, s);
}

void show(uint64 len, uint64 *d_data) {
  uint64 *h_data = new uint64[len];
  cudaMemcpy(h_data, d_data, sizeof(uint64) * len, cudaMemcpyDeviceToHost);
  for (int i=0; i<len; i++) {
    printf("%llu ", h_data[i]);
  }
  printf("\n");
  delete[] h_data;
}

void showc(uint64 len, uint64 *data) {
  for (uint64 i=0; i<len; i++) {
    printf("%llu ", data[i]);
  }
  printf("\n");
}