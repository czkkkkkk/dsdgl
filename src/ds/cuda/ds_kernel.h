#ifndef DGL_DS_KERNEL_H_
#define DGL_DS_KERNEL_H_

#include <nccl.h>
#include <stdint.h>
#include <stdlib.h>
#include <dgl/array.h>
#include <dgl/aten/csr.h>

using namespace dgl;
using namespace dgl::aten;
using uint64 = unsigned long long int;
using IdType = uint64;

namespace dgl {
namespace ds {

const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;
const int BLOCK_NUM = 2;

/**
 * @brief (inplace) Convert global nid to local nid
 * @param global_ids      Global node ids on device
 * @param min_vids        Graph node partition scheme
 * @param rank            rank
 */
void ConvertGidToLid(IdArray global_ids, IdArray min_vids, int rank);

void ConvertLidToGid(IdArray local_ids, IdArray global_nid_map);

void Cluster(IdArray seeds, IdArray min_vids, int world_size, IdArray* send_sizes, IdArray* dev_send_offset);

void Shuffle(IdArray seeds, IdArray host_send_offset, IdArray send_sizes, int rank, int world_size, ncclComm_t nccl_comm, IdArray* frontier, IdArray* host_recv_offset);

void SampleNeighbors(IdArray frontier, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges);

void Reshuffle(IdArray neighbors, int fanout, int n_seeds, IdArray host_shuffle_send_offset, IdArray host_shuffle_recv_offset, int rank, int world_size, ncclComm_t nccl_comm, IdArray* reshuffled_neighbors);

}
}

#endif