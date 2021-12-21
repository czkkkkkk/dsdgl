#ifndef DGL_DS_KERNEL_H_
#define DGL_DS_KERNEL_H_

#include <nccl.h>
#include <stdint.h>
#include <stdlib.h>

using uint64 = unsigned long long int;

const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;
const int BLOCK_NUM = 2;

void Cluster( int num_devices, uint64 *device_vid_base,
              uint64 num_seed, uint64 *seeds, int fanout,
              uint64 *device_col_ptr, uint64 *device_col_cnt, uint64 *d_device_col_cnt);

void Shuffle(int num_devices, uint64 *device_col_ptr, uint64 *device_col_cnt, uint64 *d_device_col_cnt, 
             uint64 *seeds, uint64 &num_frontier, uint64 *device_offset, uint64 **frontier, 
             int rank, ncclComm_t &comm);

void SampleNeighbors();

void Reshuffle();

void show(uint64 len, uint64 *d_data);

void showc(uint64 len, uint64 *data);

#endif