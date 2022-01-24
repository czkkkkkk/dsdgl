#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <nccl.h>
#include <thread>
#include <numeric>

#include "ds/cuda/scan.h"
#include "dmlc/logging.h"
#include "./test_utils.h"

using namespace dgl;
using namespace dgl::runtime;
using namespace dgl::ds;

IdArray GetIdArray(const std::vector<IdType>& vec) {
  return IdArray::FromVector(vec, DLContext{kDLGPU, 0});
}

std::vector<IdType> RandVec(int size, int max_val) {
  std::vector<IdType> ret(size);
  for(int i = 0; i < size; ++i) {
    ret[i] = rand() % max_val;
  }
  return ret;
}

std::pair<std::vector<IdType>, std::vector<IdType>>
_Scan(const std::vector<IdType>& seeds, const std::vector<IdType>& part_offset, const std::vector<IdType>& part_ids, int world_size) {
  IdArray sorted, index;
  std::tie(sorted, index) = MultiWayScan(GetIdArray(seeds), GetIdArray(part_offset), GetIdArray(part_ids), world_size);
  return {sorted.ToVector<IdType>(), index.ToVector<IdType>()};
}

std::pair<std::vector<IdType>, std::vector<IdType>>
_NaiveScan(const std::vector<IdType>& seeds, const std::vector<IdType>& part_offset, const std::vector<IdType>& part_ids, int world_size) {
  std::vector<IdType> sorted(seeds.size()), index(seeds.size()), count(world_size, 0);
  for(int i = 0; i < seeds.size(); ++i) {
    int r = part_ids[i];
    int c = count[r]++;
    int p = c + part_offset[r];
    sorted[p] = seeds[i];
    index[p] = i;
  }
  return {sorted, index};
}


std::vector<IdType> GetOffset(const std::vector<IdType>& part_ids, int world_size) {
  std::vector<IdType> ret(world_size + 1, 0);
  for(auto id: part_ids) {
    ret[id+1] += 1;
  }
  for(int i = 1; i <= world_size; ++i) {
    ret[i] += ret[i-1];
  }
  return ret;
}

void _TestScan(const std::vector<IdType>& seeds, const std::vector<IdType>& part_ids, int world_size) {
  auto part_offset = GetOffset(part_ids, world_size);
  std::vector<IdType> sorted, index, exp_sorted, exp_index;
  std::tie(sorted, index) = _Scan(seeds, part_offset, part_ids, world_size);
  std::tie(exp_sorted, exp_index) = _NaiveScan(seeds, part_offset, part_ids, world_size);
  CheckVectorEq(sorted, exp_sorted);
  CheckVectorEq(index, exp_index);

}
TEST(DSSampling, Scan) {
  int world_size = 3, n_seeds = 4;
  _TestScan({1, 3, 2, 4}, {0, 2, 1, 1}, world_size);
  world_size = 1, n_seeds = 10;
  _TestScan(RandVec(n_seeds, 100), RandVec(n_seeds, world_size), world_size);
  world_size = 3, n_seeds = 1024;
  _TestScan(RandVec(n_seeds, 100), RandVec(n_seeds, world_size), world_size);
}

TEST(DSSampling, ScanLarge) {
  int world_size, n_seeds;
  world_size = 3, n_seeds = 1025;
  _TestScan(RandVec(n_seeds, 100), RandVec(n_seeds, world_size), world_size);
  world_size = 2, n_seeds = 100020;
  _TestScan(RandVec(n_seeds, 100), RandVec(n_seeds, world_size), world_size);
}