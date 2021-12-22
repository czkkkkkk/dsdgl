#ifndef DGL_DS_KERNEL_H_
#define DGL_DS_KERNEL_H_

#include <nccl.h>
#include <stdint.h>
#include <stdlib.h>

using uint64 = unsigned long long int;

const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;
const int BLOCK_NUM = 2;

/**
 * @brief convert global id to local id in each partition
 * @param num_id          the size of global_id
 * @param global_id       the global id array
 * @param d_device_vids   the start id on each device
 * @param rank            current device/partition
 */
void ConvertGidToLid(uint64 num_id, uint64 *global_id, uint64 *d_device_vids, int rank);

/**
 * @brief convert local id to global id in each partition
 * @param num_id          the size of local
 * @param local_id        the local id array
 * @param d_device_vids   the start id on each device
 * @param rank            current device/partition
 */
void ConvertLidToGid(uint64 num_id, uint64 *local_id, uint64 *d_device_vids, int rank);

/**
 * @brief sort seeds and cluster seeds according to their partition
 * @param num_devices       the number of devices
 * @param device_vid_base   the start vid of each device
 * @param num_seed          the number of seeds
 * @param seeds             seeds array
 * @param device_col_ptr    the offset array of seeds clusters
 * @param device_col_cnt    the size array of clusters of each device on host
 * @param d_device_col_cnt  the size array of clusters of each device on device
 */
void Cluster(int num_devices, uint64 *device_vid_base,
             uint64 num_seed, uint64 *seeds,
             uint64 *device_col_ptr, uint64 *device_col_cnt, 
             uint64 **d_device_col_cnt);

/**
 * @brief send seed clusters to corresponding devices
 * @param num_devices       the number of devices
 * @param device_col_ptr    the offset array of clusters
 * @param device_col_cnt    the size array of clusters of each device on host
 * @param d_device_col_cnt  the size array of clusters of each device on device
 * @param seeds             seeds array to send to each devices
 * @param num_frontier      received seeds number
 * @param device_offset     the offsets of each received seeds cluster from another device
 * @param frontier          received seeds
 * @param rank              current device
 * @param comm              nccl comm
 */
void Shuffle(int num_devices, uint64 *device_col_ptr, uint64 *device_col_cnt, uint64 *d_device_col_cnt, 
             uint64 *seeds, uint64 &num_frontier, uint64 *device_offset, uint64 **frontier, 
             int rank, ncclComm_t &comm);

/**
 * @brief do local sampling, using replace sampling so the result size is determined
 * @param fanout            the number of neighbor to sample
 * @param num_frontier      the number vertices to sample
 * @param frontier          vertices to sample
 * @param in_ptr            graph csr rows
 * @param in_index          graph csr cols
 * @param edge_index        graph edge idx (maybe)
 * @param out_index         the cols of the output csr
 * @param out_edges         the edge indices of the output csr
 */
void SampleNeighbors(int fanout, uint64 num_frontier, uint64 *frontier,
                     uint64 *in_ptr, uint64 *in_index, uint64 *edge_index,
                     uint64 **out_index, uint64 **out_edges);

/**
 * @brief send the sampled results back to the original devices
 * @param fanout            the number of neighbor to sample
 * @param num_devices       the number of devices
 * @param device_offset     the offsets of each received seeds cluster from another device
 * @param cols              sampled cols results to send
 * @param edges             sampled edge indices to send
 * @param device_col_ptr    the offset array of clusters
 * @param num_seed          number of seeds (not frontier but the original batch)
 * @param out_ptr           the rows of the received csr
 * @param out_cols          the cols of the received csr
 * @param out_edges         the edge indices of the received csr
 * @param rank              current device
 * @param comm              nccl comm
 */
void Reshuffle(int fanout, int num_devices, uint64 *device_offset, uint64 *cols, uint64 *edges,
               uint64 *device_col_ptr, uint64 num_seed, uint64 **out_ptr, 
               uint64 **out_cols, uint64 **out_edges,
               int rank, ncclComm_t &comm);

void show(uint64 len, uint64 *d_data);

void showc(uint64 len, uint64 *data);

void PingPong(int rank, ncclComm_t &comm);

#endif