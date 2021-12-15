#ifndef SAMPLER_H
#define SAMPLER_H

#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "nccl.h"
#include "mpi.h"
#include <iostream>

const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;
const int BLOCK_NUM = 2;

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
void countDeviceVerticesKernel(int device_cnt, int *device_vid_base,
                           int num_seed, int *seeds, int num_picks, 
                           int *device_col_cnt) {
  __shared__ int local_count[9];
  __shared__ int device_vid[9];
  assert(blockDim.x == BLOCK_SIZE);
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (threadIdx.x <= device_cnt) {
    device_vid[threadIdx.x] = device_vid_base[threadIdx.x];
    local_count[threadIdx.x] = 0;
  }

  __syncthreads();

  int device_id, vid;
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

void init(int *&d_row_ptr, int *&d_cols, int num_seed, int fanout) {
  cudaMalloc((void**)&d_row_ptr, sizeof(int)*(num_seed + 1));
  thrust::device_ptr<int> d_d_row_ptr(d_row_ptr);
  thrust::fill(d_d_row_ptr, d_d_row_ptr + num_seed, fanout);
  thrust::exclusive_scan(d_d_row_ptr, d_d_row_ptr + num_seed + 1, d_row_ptr);
  cudaMalloc((void**)&d_cols, sizeof(int)*num_seed*fanout);
  memset(d_cols, 0, sizeof(int)*num_seed*fanout);
}

void cluster(int device_cnt, 
             int *device_vid_base, //每个device上的其实vid
             int num_seed, int *&seeds, int fanout,
             int *device_col_ptr,  //这些seed里每个device的其实位置在哪里 
             int *device_col_cnt   //这些seed里每个device有多少个
            ) {
  thrust::device_ptr<int> d_seeds(seeds);
  thrust::sort(d_seeds, d_seeds + num_seed);
  
  int *d_device_col_ptr, *d_device_col_cnt;
  cudaMalloc((void**)&d_device_col_cnt, sizeof(int)*device_cnt);
  cudaMemset(d_device_col_cnt, 0, sizeof(int)*device_cnt);
  cudaMalloc((void**)&d_device_col_ptr, sizeof(int)*(device_cnt+1));
  cudaMemset(d_device_col_ptr, 0, sizeof(int)*(device_cnt+1));

  countDeviceVerticesKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(device_cnt, device_vid_base,
                                                   num_seed, seeds, fanout,
                                                   d_device_col_cnt);
  
  thrust::exclusive_scan(thrust::device_ptr<int>(d_device_col_cnt), 
                         thrust::device_ptr<int>(d_device_col_cnt) + device_cnt + 1, d_device_col_ptr);  
  cudaMemcpy(device_col_ptr, d_device_col_ptr, sizeof(int)*(device_cnt+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(device_col_cnt, d_device_col_cnt, sizeof(int)*device_cnt, cudaMemcpyDeviceToHost);
  cudaFree(d_device_col_cnt);
  cudaFree(d_device_col_ptr);
}

int* convert(thrust::device_ptr<int> ptr) {
  return thrust::raw_pointer_cast(ptr);
}

template <int BLOCK_WARPS, int TILE_SIZE>
__global__
void cuSample(int graph_base, int *graph_row_ptr, int *graph_cols,
              int num_frontier, int *frontier, int fanout,
              int *row_ptr, int *cols) {
  int out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_row = min(static_cast<int>(blockIdx.x + 1) * TILE_SIZE, num_frontier);
  curandState rng;
  curand_init(blockIdx.x, threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);
  while (out_row < last_row) {
    const int row = frontier[out_row] - graph_base;
    const int in_row_start = graph_row_ptr[row];
    const int deg = graph_row_ptr[row + 1] - in_row_start;
    const int out_row_start = row_ptr[out_row];
        
    if (deg <= fanout) {
      // just copy row
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        const int in_idx = in_row_start + idx;
        cols[out_row_start + idx] = graph_cols[in_idx];
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < fanout; idx += WARP_SIZE) {
        cols[out_row_start + idx] = idx;
      }
      __syncwarp();
        
      for (int idx = fanout + threadIdx.x; idx < deg; idx += WARP_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < fanout) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          atomicMax(reinterpret_cast<int *>(cols + out_row_start + num), idx);
        }
      }
      __syncwarp();
        
      // copy permutation over
      for (int idx = threadIdx.x; idx < fanout; idx += WARP_SIZE) {
        const int perm_idx = cols[out_row_start + idx] + in_row_start;
        cols[out_row_start + idx] = graph_cols[perm_idx];
      }

    }
    out_row += BLOCK_WARPS;
  }
}

//假设节点id>0，那么0可以表示没采满
void sample(int graph_base, int *graph_row_ptr, int *graph_cols,
            int num_frontier, int *frontier, int fanout,
            int *&cols) {
  int *row_ptr; 
  cudaMalloc((void**)&row_ptr, sizeof(int)*(num_frontier+1));
  thrust::device_ptr<int> d_row_ptr(row_ptr);
  thrust::fill(d_row_ptr, d_row_ptr + num_frontier, fanout);
  thrust::exclusive_scan(d_row_ptr, d_row_ptr + num_frontier + 1, d_row_ptr);
  cudaMalloc((void**)&cols, sizeof(int)*num_frontier*fanout);
  cudaMemset(cols, 0, sizeof(int)*num_frontier*fanout);

  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((num_frontier + TILE_SIZE - 1) / TILE_SIZE);
  
  cuSample<BLOCK_WARPS, TILE_SIZE><<<grid, block>>>(graph_base, graph_row_ptr, graph_cols,
                                                    num_frontier, frontier, fanout,
                                                    row_ptr, cols);
  cudaFree(row_ptr);
}

void reshuffle(int fanout,
               int device_cnt, int *device_offset, int *cols,
               int *device_col_ptr, int *clustered_cols,
               int rank, ncclComm_t &comm, cudaStream_t &s) {
  thrust::device_ptr<int> d_cols(cols), d_clustered_cols(clustered_cols);
  CUDACHECK(cudaMemcpy(convert(d_clustered_cols + device_col_ptr[rank] * fanout), 
                       convert(d_cols + device_offset[rank] * fanout),
                       (device_offset[rank + 1] - device_offset[rank]) * fanout * sizeof(int), 
                       cudaMemcpyDeviceToDevice));
  for (int i=0; i<device_cnt; i++) {
    if (i == rank) {
      for (int j=0; j<device_cnt; j++) {
        if (j != rank) {
          NCCLCHECK(ncclSend((const void*)convert(d_cols + device_offset[j] * fanout), 
                             (device_offset[j + 1] - device_offset[j]) * fanout, 
                             ncclInt, j, comm, s));
        }
      }
    } else {
        NCCLCHECK(ncclRecv((void*)convert(d_clustered_cols + device_col_ptr[i] * fanout), 
                           (device_col_ptr[i + 1] - device_col_ptr[i]) * fanout, 
                           ncclInt, i, comm, s));
    }
  }
  CUDACHECK(cudaStreamSynchronize(s));
}

#endif