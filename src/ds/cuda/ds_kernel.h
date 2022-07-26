#ifndef DGL_DS_KERNEL_H_
#define DGL_DS_KERNEL_H_

#include <nccl.h>
#include <stdint.h>
#include <stdlib.h>
#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include "../../runtime/cuda/cuda_common.h"
#include "cuda_utils.h"

using namespace dgl;
using namespace dgl::aten;

using IdType = int64_t;
using DataType = float;

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

std::tuple<IdArray, IdArray, IdArray, IdArray> Partition(IdArray seeds, IdArray min_vids, int world_size);

IdArray Partition(IdArray seeds, IdArray min_vids);

void Cluster(int rank, IdArray seeds, IdArray min_vids, int world_size, IdArray* send_sizes, IdArray* send_offset);

IdArray SampleNeighbors(IdArray frontier, int fanout, IdArray weight, bool bias);

void SampleNeighborsV2(IdArray frontier, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges);

void Reshuffle(IdArray neighbors, int fanout, int n_seeds, IdArray host_shuffle_send_offset, IdArray host_shuffle_recv_offset, int rank, int world_size, ncclComm_t nccl_comm, IdArray* reshuffled_neighbors, bool is_sample);

void ReshuffleV2(IdArray neighbors, int fanout, IdArray host_shuffle_recv_offset, int rank, int world_size, IdArray* reshuffled_neighbors);

IdArray Remap(IdArray neighbors, IdArray index, int fanout);

void Replicate(IdArray src, IdArray *des, int fanout);

void SampleNeighborsUVA(IdArray frontier, IdArray row_idx, CSRMatrix csr_mat, int fanout, IdArray* neighbors, IdArray* edges);

/**
 * @brief  IndexSelect is used for loading subtensors, which can handle different cases. The logic of loading subtensors can be seen as we load some rows from the input table and put them into the output table, where sometimes the input or output indices could go through some mappings.
 * 
 * If index is NullArray, the input of the i-th row is input_table[i]. Otherwise it is input_table[index[i]].
 * If input_mapping is NullArray, the input of the i-th row is input_table[index[i]]. Otherwise it is input_table[input_mapping[index[i]]].
 * If output_mapping is NullArray, the output of the i-th row is saved to output_table[i]; Otherwise it is saved to output_table[output_mapping[i]].
 * 
 * @param size The number of rows to load.
 * @param index 
 * @param input_table 
 * @param output_table 
 * @param feat_dim 
 * @param input_mapping 
 * @param output_mapping 
 * @param stream 
 */
void IndexSelect(IdType size, IdArray index, IdArray input_table, IdArray output_table, int feat_dim, IdArray input_mapping = NullArray(), IdArray output_mapping = NullArray(), cudaStream_t stream = 0);

void IndexSelectUVA(IdType size, IdArray index, IdArray input_table, IdArray output_table, int feat_dim, IdArray input_mapping = NullArray(), IdArray output_mapping = NullArray(), cudaStream_t stream = 0);

std::tuple<IdArray, IdArray, IdArray> GetFeatTypePartIds(IdArray nodes, IdArray feat_pos_map);

}
}

#endif
