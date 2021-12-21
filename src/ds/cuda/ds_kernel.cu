#include "ds_kernel.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
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
              uint64 *device_col_ptr, uint64 *device_col_cnt, uint64 **d_device_col_cnt) {
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

void Scatter(thrust::device_ptr<uint64> send_buffer, uint64 *send_offset, uint64 *send_cnt,
             thrust::device_ptr<uint64> recv_buffer, uint64 *recv_offset, uint64 *recv_cnt,
             int group_size, int rank, ncclComm_t &comm, cudaStream_t &s) {
  assert(send_cnt[rank] == recv_cnt[rank]);
  CUDACHECK(cudaMemcpy(Convert(recv_buffer + recv_offset[rank]), 
                       Convert(send_buffer + send_offset[rank]),
                       send_cnt[rank] * sizeof(uint64), cudaMemcpyDeviceToDevice));
  for (int i = 0; i < group_size; i++) {
    if (i == rank) {
      for (int j = 0; j < group_size; j++) {
        if (j != rank) {
          NCCLCHECK(ncclSend((const void*)Convert(send_buffer + send_offset[j]), 
                             send_cnt[j],
                             ncclUint64, j, comm, s));
        }
      }
    } else {
      NCCLCHECK(ncclRecv((void*)Convert(recv_buffer + recv_offset[i]), 
                          recv_cnt[i], 
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
  uint64 seq[device_cnt], ones[device_cnt];
  for (int i=0; i<device_cnt; i++) {
    seq[i] = i;
    ones[i] = 1;
  }
  Scatter(d_d_device_col_cnt, seq, ones,
          d_d_device_recv_cnt, seq, ones,
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

  Scatter(d_seeds, device_col_ptr, device_col_cnt,
          d_frontier, device_offset, device_recv_cnt,
          device_cnt, rank, comm, s);
}

template<typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    IdType * const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x*blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row+1]-in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows-1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

template<typename IdType, int BLOCK_ROWS>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == WARP_SIZE);

  // we need one state per 256 threads
  constexpr int NUM_RNG = ((WARP_SIZE*BLOCK_ROWS)+255)/256;
  __shared__ curandState rng_array[NUM_RNG];
  assert(blockDim.x >= NUM_RNG);
  if (threadIdx.y == 0 && threadIdx.x < NUM_RNG) {
    curand_init(rand_seed, 0, threadIdx.x, rng_array+threadIdx.x);
  }
  __syncthreads();
  curandState * const rng = rng_array+((threadIdx.x+WARP_SIZE*threadIdx.y)/256);

  int64_t out_row = blockIdx.x*blockDim.y+threadIdx.y;
  while (out_row < num_rows) {
    const int64_t row = in_rows[out_row];

    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];

    const int64_t deg = in_ptr[row+1] - in_row_start;

    // each thread then blindly copies in rows
    for (int idx = threadIdx.x; idx < num_picks; idx += blockDim.x) {
      const int64_t edge = curand(rng) % deg;
      const int64_t out_idx = out_row_start+idx;
      out_rows[out_idx] = row;
      out_cols[out_idx] = in_index[in_row_start+edge];
      out_idxs[out_idx] = data ? data[in_row_start+edge] : in_row_start+edge;
    }
    out_row += gridDim.x*blockDim.y;
  }
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

// int main() {
//   uint64 num_devices = 8;
//   uint64 *device_vids = new uint64[9];
//   for (int i=0; i<num_devices; i++) {
//     device_vids[i] = i;
//   }
//   uint64 *d_device_vids;
//   cudaMalloc((void**)&d_device_vids, sizeof(uint64)*num_devices);
//   cudaMemcpy(d_device_vids, device_vids, sizeof(uint64)*num_devices, cudaMemcpyHostToDevice);

//   uint64 num_seeds = 10;
//   uint64 *seeds = new uint64[num_seeds];
//   for (int i=0; i<num_seeds; i++) {
//     seeds[i] = num_seeds - i;
//   }
//   uint64 *d_seeds;
//   cudaMalloc((void**)&d_seeds, sizeof(uint64)*num_seeds);
//   cudaMemcpy(d_seeds, seeds, sizeof(uint64)*num_seeds, cudaMemcpyHostToDevice);
//   uint64 *device_col_ptr = new uint64[num_devices + 1]; 
//   uint64 *device_col_cnt = new uint64[num_devices];

//   int fanout = 2;
//   show(num_seeds, d_seeds);
//   Cluster(num_devices, d_device_vids, num_seeds, d_seeds, fanout, device_col_ptr, device_col_cnt);
//   show(num_seeds, d_seeds);
//   return 0;
// }