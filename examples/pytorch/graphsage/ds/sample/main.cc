#include <stdio.h>
#include "cuda_runtime.h"
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "mpi.h"
#include <iostream>
#include "nccl.h"

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

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

using namespace std;

int* convert(thrust::device_ptr<int> ptr);

void init(int *&d_row_ptr, int *&d_cols, int num_seed, int fanout);

void cluster(int device_cnt, 
             int *device_vid_base, //每个device上的其实vid
             int num_seed, int *&seeds, int fanout,
             int *device_col_ptr,  //这些seed里每个device的其实位置在哪里 
             int *device_col_cnt   //这些seed里每个device有多少个
            );

void shuffle(int device_cnt, int *device_col_ptr, int *device_col_cnt, int *seeds,
             int &num_frontier, int *device_offset, int *&frontier, 
             int rank, ncclComm_t &comm, cudaStream_t &s) {
  int device_recv_cnt[device_cnt];
  for (int i=0; i<device_cnt; i++) {
    MPI_Scatter(device_col_cnt, 1, MPI_INT, device_recv_cnt + i, 1, MPI_INT, i, MPI_COMM_WORLD);
  }
  device_offset[0] = 0;
  for (int i=1; i<=device_cnt; i++) {
    device_offset[i] = device_offset[i-1] + device_recv_cnt[i-1];
  }
  num_frontier = device_offset[device_cnt];
  // printf("%d\n", num_frontier);
  CUDACHECK(cudaMalloc((void**)&frontier, sizeof(int)*num_frontier));

  thrust::device_ptr<int> d_seeds(seeds), d_frontier(frontier);
  assert(device_offset[rank+1] - device_offset[rank] == device_col_cnt[rank]);
  CUDACHECK(cudaMemcpy(convert(d_frontier + device_offset[rank]), convert(d_seeds + device_col_ptr[rank]),
            device_col_cnt[rank]*sizeof(int), cudaMemcpyDeviceToDevice));
  for (int i=0; i<device_cnt; i++) {
    if (i == rank) {
      for (int j=0; j<device_cnt; j++) {
        if (j != rank) {
          NCCLCHECK(ncclSend((const void*)convert(d_seeds + device_col_ptr[j]), device_col_cnt[j], ncclInt, j, comm, s));
        }
      }
    } else {
      NCCLCHECK(ncclRecv((void*)convert(d_frontier + device_offset[i]), device_recv_cnt[i], ncclInt, i, comm, s));
    }
  }
  CUDACHECK(cudaStreamSynchronize(s));
}

void reshuffle(int fanout,
               int device_cnt, int *device_offset, int *cols,
               int *device_col_ptr, int *clustered_cols,
               int rank, ncclComm_t &comm, cudaStream_t &s);

void sample(int graph_base, int *graph_row_ptr, int *graph_cols,
            int num_frontier, int *frontier, int fanout,
            int *&cols);

void neighborSampling(int *d_graph_row_ptr, int *d_graph_cols,
                      int device_cnt, int *d_device_vid_base,
                      int num_seed, int *d_seeds, int fanout,
                      int rank, ncclComm_t &comm, cudaStream_t &s,
                      int *&d_row_ptr, int *&d_cols) {
  init(d_row_ptr, d_cols, num_seed, fanout);
  
  int h_device_seed_ptr[device_cnt + 1], h_device_seed_cnt[device_cnt];
  cluster(device_cnt, d_device_vid_base, 
          num_seed, d_seeds, fanout,
          h_device_seed_ptr, h_device_seed_cnt);
  
  int num_frontier;
  int *d_frontier;
  int h_device_offset[device_cnt + 1];
  shuffle(device_cnt, h_device_seed_ptr, h_device_seed_cnt, d_seeds,
          num_frontier, h_device_offset, d_frontier,
          rank, comm, s);
  
  int h_device_vid_base[device_cnt];
  cudaMemcpy(h_device_vid_base, d_device_vid_base, sizeof(int)*device_cnt, cudaMemcpyDeviceToHost);
  int *d_tmp_cols;
  sample(h_device_vid_base[rank], d_graph_row_ptr, d_graph_cols,
         num_frontier, d_frontier, fanout,
         d_tmp_cols);
  
  reshuffle(fanout,
            device_cnt, h_device_offset, d_tmp_cols,
            h_device_seed_ptr, d_cols,
            rank, comm, s);
  
  cudaFree(d_frontier);
  cudaFree(d_tmp_cols);
}

// int main(int argc, char *argv[]) {
//   int rank, nRanks;  
//   //initializing MPI
//   MPICHECK(MPI_Init(&argc, &argv));
//   MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
//   MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

//   ncclUniqueId id;
//   ncclComm_t comm;
//   if (rank == 0) {
//     ncclGetUniqueId(&id);
//   }
//   MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

//   CUDACHECK(cudaSetDevice(rank));
//   NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, rank));
//   cudaStream_t s;
//   CUDACHECK(cudaStreamCreate(&s));
  
//   int fanout = 3;
//   int device_offset[nRanks + 1];
//   int cols[nRanks * fanout];
//   for (int i=0; i<=nRanks; i++) {
//     device_offset[i] = i;
//     cout<<device_offset[i]<<" ";
//   }
//   cout<<endl;
//   for (int i=0; i<nRanks; i++) {
//     for (int j=device_offset[i]*fanout; j<(device_offset[i]+1)*fanout; j++) {
//       cols[j] = i;
//       cout<<cols[j]<<" ";
//     }
//   }
//   cout<<endl;
//   int *d_cols;
//   cudaMalloc((void**)&d_cols, sizeof(int)*(nRanks * fanout));
//   CUDACHECK(cudaMemcpy(d_cols, cols, sizeof(int)*(nRanks * fanout), cudaMemcpyHostToDevice));

//   int *clustered_cols;
//   cudaMalloc((void**)&clustered_cols, sizeof(int)*(nRanks * fanout));

//   int device_ptr[nRanks + 1];
//   for (int i=0; i<=nRanks; i++) {
//     device_ptr[i] = i;
//   }
//   reshuffle(fanout, nRanks, device_offset, d_cols, device_ptr, clustered_cols, rank, comm, s);

//   int h_c[nRanks * fanout];
//   cudaMemcpy(h_c, clustered_cols, sizeof(int)*(nRanks * fanout), cudaMemcpyDeviceToHost);

//   cout<<"device: "<<rank<<endl;
//   for (int i=0; i<nRanks * fanout; i++) {
//     cout<<h_c[i]<<" ";
//   }
//   cout<<endl;

//   ncclCommDestroy(comm);
//   MPICHECK(MPI_Finalize());
//   return 0;
// }

// int main() {
//   int row[10], cols[10*5];
//   int total = 0;
//   for (int i=0; i<10; i++) {
//     row[i] = total;
//     for (int j=0; j<5; j++) {
//       cols[total + j] = i + j + 1;
//     }
//     total += 5;
//   }
//   int *d_row, *d_cols;
//   cudaMalloc(&d_row, sizeof(int)*10);
//   cudaMemcpy(d_row, row, sizeof(int)*10, cudaMemcpyHostToDevice);
//   cudaMalloc(&d_cols, sizeof(int)*10*5);
//   cudaMemcpy(d_cols, cols, sizeof(int)*10*5, cudaMemcpyHostToDevice);

//   int num_seed = 3;
//   int seeds[num_seed];
//   for (int i=0; i<num_seed; i++) {
//     seeds[i] = i;
//   }
//   int *d_seeds;
//   cudaMalloc((void**)&d_seeds, sizeof(int)*num_seed);
//   int *out_cols;
//   int fanout = 6;
//   sample(0, d_row, d_cols, num_seed, d_seeds, fanout, out_cols);

//   int h_cols[num_seed*fanout];
//   cudaMemcpy(h_cols, out_cols, sizeof(int)*num_seed*fanout, cudaMemcpyDeviceToHost);
//   for (int i=0; i<num_seed*fanout; i++) {
//     cout<<h_cols[i]<<" ";
//   }
//   cout<<endl;

//   return 0;
// }

// int main(int argc, char *argv[]) {
//   int rank, nRanks;  
//   //initializing MPI
//   MPICHECK(MPI_Init(&argc, &argv));
//   MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
//   MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

//   ncclUniqueId id;
//   ncclComm_t comm;
//   if (rank == 0) {
//     ncclGetUniqueId(&id);
//   }
//   MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

//   CUDACHECK(cudaSetDevice(rank));
//   NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, rank));
//   cudaStream_t s;
//   CUDACHECK(cudaStreamCreate(&s));

//   int num_seeds = (1+nRanks)*nRanks/2, pre = 0;
//   int seeds[num_seeds];
//   for (int i=0; i<nRanks; i++) {
//     for (int j=0; j<=i; j++) {
//       seeds[pre + j] = i;
//     }
//     pre += i + 1;
//   }
//   int *d_seeds;
//   cudaMalloc((void**)&d_seeds, sizeof(int)*num_seeds);
//   cudaMemcpy(d_seeds, seeds, sizeof(int)*num_seeds, cudaMemcpyHostToDevice);

//   int total = 0;
//   int device_col_cnt[nRanks], device_col_ptr[nRanks];
//   for (int i=0; i<nRanks; i++) {
//     device_col_cnt[i] = i + 1;
//     device_col_ptr[i] = total;
//     total += device_col_cnt[i];
//   }
//   int num_frontier, *frontier;
//   int device_offset[nRanks+1];

//   shuffle(nRanks, device_col_ptr, device_col_cnt, d_seeds,
//           num_frontier, device_offset, frontier, 
//           rank, comm, s);

//   cout<<"device: "<<rank<<endl;
//   int h_f[num_frontier];
//   cudaMemcpy(h_f, frontier, sizeof(int)*num_frontier, cudaMemcpyDeviceToHost);
//   for (int i=0; i<num_frontier; i++) {
//     cout<<h_f[i]<<" ";
//   }
//   cout<<endl;

//   ncclCommDestroy(comm);
//   MPICHECK(MPI_Finalize());
//   return 0;
// }

// int main() {
//   int num_seed = 1024;
//   int *seeds = new int[num_seed];
//   for (int i=0; i<num_seed; i++) {
//     seeds[i] = num_seed - i - 1;
//   }

//   int *d_seeds = nullptr;
//   CUDACHECK(cudaMalloc((void**)&d_seeds, sizeof(int)*num_seed));
//   CUDACHECK(cudaMemcpy(d_seeds, seeds, sizeof(int)*num_seed, cudaMemcpyHostToDevice));

//   const int device_cnt = 4;
//   int device_vid_base[device_cnt] = {0};
//   for (int i=0; i<device_cnt; i++) {
//     device_vid_base[i] = (num_seed / device_cnt) * i;
//   }
//   device_vid_base[device_cnt] = num_seed;

//   for (int i=0; i<=device_cnt; i++) {
//     printf("%d ", device_vid_base[i]);
//   }
//   printf("\n");

//   int *d_device_vid_base = nullptr;
//   CUDACHECK(cudaMalloc((void**)&d_device_vid_base, sizeof(int)*(device_cnt+1)));
//   CUDACHECK(cudaMemcpy(d_device_vid_base, device_vid_base, sizeof(int)*(device_cnt+1), cudaMemcpyHostToDevice));

//   int num_picks = 10;
//   int device_col_ptr[device_cnt + 1];
//   int device_col_cnt[device_cnt + 1];
//   cluster(device_cnt, d_device_vid_base, num_seed, d_seeds, num_picks, device_col_ptr, device_col_cnt);

//   for (int i=0; i<=device_cnt; i++) {
//     printf("%d\n", device_col_cnt[i]);
//     printf("%d\n", device_col_ptr[i]);
//   }

//   return 0;
// }